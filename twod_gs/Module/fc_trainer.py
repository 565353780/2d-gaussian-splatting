import os
import sys
import torch
import open3d as o3d

from torch import nn
from typing import Tuple
from copy import deepcopy
from tqdm import tqdm, trange
from argparse import ArgumentParser

from fused_ssim import fused_ssim

from utils.general_utils import inverse_sigmoid
from utils.mesh_utils import GaussianExtractor, post_process_mesh

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Method.general_utils import colormap
from base_gs_trainer.Loss.chamfer import chamferLossFn
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from flexi_cubes.Module.fc_convertor import FCConvertor

from mv_fc_recon.Loss.mesh_geo_energy import thin_plate_energy

from twod_gs.Config.config import ModelParams, PipelineParams, OptimizationParams
from twod_gs.Method.render_kernel import render
from twod_gs.Method.fast_utils import sampling_cameras, compute_gaussian_score_fastgs
from twod_gs.Model.gs import GaussianModel


class FCTrainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        init_mesh_file_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=100,
        save_freq: int=100,
        fc_update_freq: int=1,
        log_start_iter: int=0,
    ) -> None:
        self.fc_update_freq = fc_update_freq
        self.log_start_iter = log_start_iter

        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(sys.argv[1:])

        args.source_path = colmap_data_folder_path
        args.model_path = save_result_folder_path

        print("Optimizing " + args.model_path)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)

        BaseGSTrainer.__init__(
            self,
            colmap_data_folder_path=colmap_data_folder_path,
            device=device,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            test_freq=test_freq,
            save_freq=save_freq,
        )

        # Ray-wise Surface Winner + Surface hardening (Phase B)
        # S_i = fraction of rays where Gaussian i is winner (from winner_id); C_i = EMA(S_i)
        self.surface_confidence = None  # shape (N,) when inited; EMA of S_i
        self.surface_ema_momentum = getattr(self.opt, "surface_ema_momentum", 0.99)
        self.surface_hardening_start_iter = getattr(self.opt, "surface_hardening_start_iter", 10_000)
        # Phase B：仅奖励表面附近 Gaussian 的 opacity→1，非表面只做轻微惩罚；loss 均做归一化避免压倒 rgb
        self.lambda_surface = getattr(self.opt, "lambda_surface", 0.05)
        self.lambda_exclusive = getattr(self.opt, "lambda_exclusive", 0.01)  # 轻微惩罚 loser，远小于 λ_surface
        self.surface_confidence_max = getattr(self.opt, "surface_confidence_max", 0.1)  # clamp C_i 避免梯度压倒 photometric
        self.surface_confidence_min = getattr(self.opt, "surface_confidence_min", 0.01)  # C_i 低于此不施加 push→1
        self.lambda_winner_opacity = getattr(self.opt, "lambda_winner_opacity", 0.05)  # 鼓励 winner gaussians 的 opacity→1

        assert os.path.exists(init_mesh_file_path)
        self.fc_params = FCConvertor.createFC(
            init_mesh_file_path,
            resolution=192,
            device=self.device,
        )

        self.extractMesh()[0].export(self.save_result_folder_path + 'start_fc_mesh.ply')

        lr_sdf  = 1e-3
        lr_deform = 1e-3
        lr_weight = 1e-3
        param_groups = [
            dict(params=[self.fc_params['sdf']], lr=lr_sdf),
            dict(params=[self.fc_params['deform']], lr=lr_deform),
            dict(params=[self.fc_params['weight']], lr=lr_weight),
        ]

        self.fc_optimizer = torch.optim.Adam(param_groups)

        self.chamfer_func = chamferLossFn(self.device)

        self.E_thinplate_base = None
        return

    def renderImage(self, viewpoint_cam) -> dict:
        return render(viewpoint_cam, self.gaussians, self.pipe, self.background)

    def extractMesh(self):
        current_mesh, vertices, L_dev = FCConvertor.extractMesh(self.fc_params, training=True)
        return current_mesh, vertices, L_dev

    def trainStep(
        self,
        iteration: int,
        viewpoint_cam,
        lambda_dssim: float = 0.2,
        lambda_normal: float = 0.01,
        lambda_dist: float = 0.01,
        lambda_opacity: float = 0.01,
        lambda_scaling: float = 1.0,
        lambda_surf_scaling: float = 1.0,
    ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        render_pkg = self.renderImage(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        rgb_loss = (1.0 - lambda_dssim) * reg_loss + lambda_dssim * ssim_loss

        lambda_normal = lambda_normal if iteration > 1000 else 0.0
        normal_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_normal > 0:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_dot = torch.sum(rend_normal * surf_normal, dim=0)

            valid_dot_idxs = torch.where(normal_dot != 0)
            valid_normal_dot = normal_dot[valid_dot_idxs]

            normal_error = (1 - valid_normal_dot)
            normal_loss = lambda_normal * normal_error.mean()

        lambda_dist = lambda_dist if iteration > 1000 else 0.0
        dist_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            valid_dist_idxs = torch.where(rend_dist != 0)
            valid_rend_dist = rend_dist[valid_dist_idxs]
            dist_loss = lambda_dist * valid_rend_dist.mean()

        # Phase A: Surface selection — push non-surface to 0 (opacity, scale)；Phase B 开始后彻底关闭
        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0:
            opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0:
            scaling_loss = lambda_scaling * nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # Phase B: Surface hardening — ray-wise exclusivity + temporal stability
        # winner_id = argmax_i (α_i · T_before_i). S_i = n_winner_i / (n_hit_i + ε)；n_hit_i 来自 rasterizer hit_counts，
        # 仅在有 Gaussian 参与 blending 的 pixel 上累加，winner_id=-1（背景）的 pixel 不计入任何 n_hit_i，故 S_i 分母已排除背景。
        # C_i = EMA(S_i)。L_surface / L_exclusive 见下。
        surface_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        exclusive_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        winner_opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        surf_scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)

        if iteration > self.surface_hardening_start_iter:
            winner_id = render_pkg.get("winner_id")
            hit_counts = render_pkg.get("hit_counts")
            if winner_id is not None and hit_counts is not None:
                N = self.gaussians.get_xyz.shape[0]
                H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
                winner_flat = winner_id.reshape(-1).long().to(rgb_loss.device)
                n_hit = hit_counts.float().to(rgb_loss.device)
                eps = 1e-8
                valid = winner_flat >= 0
                with torch.no_grad():  # EMA 与 S_i 统计均在 no_grad 下更新
                    if valid.any():
                        indices = winner_flat[valid].clamp(max=N - 1).to(torch.int64)
                        n_winner = torch.bincount(indices, minlength=N)
                        # S_i = n_winner_i / (n_hit_i + ε): "when hit, how often is i the winner" (no area bias)
                        S_i = n_winner.float() / (n_hit + eps)
                    else:
                        n_winner = torch.zeros(N, device=rgb_loss.device, dtype=torch.float32)
                        S_i = torch.zeros(N, device=rgb_loss.device, dtype=torch.float32)
                    old_N = self.surface_confidence.shape[0] if self.surface_confidence is not None else 0
                    if self.surface_confidence is None or old_N == 0:
                        self.surface_confidence = S_i.clone()
                    elif N > old_N:
                        # N 增加（densification / split clone）：旧 index 保留 EMA，新 index C_i = 0
                        new_confidence = torch.zeros(N, device=S_i.device, dtype=S_i.dtype)
                        new_confidence[:old_N] = (
                            self.surface_ema_momentum * self.surface_confidence
                            + (1.0 - self.surface_ema_momentum) * S_i[:old_N]
                        )
                        new_confidence[old_N:] = S_i[old_N:]  # 新 Gaussian 用当前 S_i（通常为 0）
                        self.surface_confidence = new_confidence
                    elif N < old_N:
                        # N 减少（prune）：无旧→新 index 映射时保守重置
                        self.surface_confidence = S_i.clone()
                    else:
                        self.surface_confidence = (
                            self.surface_ema_momentum * self.surface_confidence
                            + (1.0 - self.surface_ema_momentum) * S_i
                        )
                # L_surface: 仅对 C_i > C_min 的“表面”Gaussian 施加 push→1，按表面点数归一化，避免随 N 爆炸
                if self.surface_confidence.shape[0] == N:
                    C_i_raw = torch.clamp(
                        self.surface_confidence.detach().to(rgb_loss.dtype),
                        0.0, self.surface_confidence_max,
                    )
                    surface_mask = (C_i_raw > self.surface_confidence_min).to(rgb_loss.dtype)
                    C_i = C_i_raw * surface_mask
                    n_surface = surface_mask.sum().clamp(min=1.0)
                    s_i = self.gaussians._opacity.squeeze(-1)
                    s_target = inverse_sigmoid(torch.tensor(0.99, device=s_i.device, dtype=s_i.dtype))
                    surface_loss = self.lambda_surface * (C_i * torch.relu(s_target - s_i).pow(2)).sum() / n_surface

                    # L_scale (surface-weighted scale collapse): 仅对 surface_mask 对应的 gaussians 计算
                    if lambda_surf_scaling > 0:
                        scale_xy = self.gaussians.get_scaling  # (N, 2)
                        scale_sq_sum = (scale_xy ** 2).sum(dim=1)  # (N,)
                        surf_scaling_loss = lambda_surf_scaling * (C_i * scale_sq_sum).sum() / n_surface
                # L_exclusive: 对 loser 做轻微惩罚，按 ray 数 (H*W) 归一化为 per-ray，避免压倒 rgb
                alpha_i = self.gaussians.get_opacity.squeeze(-1)
                n_loser_i = (n_hit - n_winner).clamp(min=0).to(rgb_loss.dtype)
                n_rays = float(H * W)
                exclusive_loss = self.lambda_exclusive * (alpha_i.pow(2) * n_loser_i).sum() / n_rays
                # L_winner_opacity: 鼓励 winner gaussians 的 opacity→1，按 n_winner 加权
                if n_winner.sum() > 0:
                    raw_winner_opacity = (n_winner * (1.0 - alpha_i).pow(2)).sum() / (n_winner.sum() + eps)
                    winner_opacity_loss = self.lambda_winner_opacity * raw_winner_opacity

        # loss
        total_loss = \
            rgb_loss + \
            dist_loss + \
            normal_loss + \
            opacity_loss + \
            scaling_loss + \
            surface_loss + \
            exclusive_loss + \
            winner_opacity_loss + \
            surf_scaling_loss

        total_loss.backward()

        if iteration > 1500 and iteration % self.fc_update_freq == 0:
            self.fc_optimizer.step()
            self.fc_optimizer.zero_grad()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'dist': dist_loss.item(),
            'normal': normal_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'surf_scaling': surf_scaling_loss.item(),
            'surface': surface_loss.item(),
            'exclusive': exclusive_loss.item(),
            'winner_opacity': winner_opacity_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    def trainFCStep(
        self,
        lambda_dev: float = 0.01,
        lambda_fc_normal: float = 1.0,
        lambda_fc_depth: float = 0.1,
        lambda_chamfer: float = 1.0,
        lambda_thin_plate: float = 0.1,
    ) -> dict:
        camera_num = len(self.scene)

        dev_loss = torch.tensor(0.0, device=self.device)
        fc_normal_loss = torch.tensor(0.0, device=self.device)
        fc_depth_loss = torch.tensor(0.0, device=self.device)
        chamfer_loss = torch.tensor(0.0, device=self.device)
        thinplate_loss = torch.tensor(0.0, device=self.device)

        fc_mesh, vertices, L_dev = self.extractMesh()

        print('[INFO][FCTrainer::trainFCStep]')
        print('\t start match fc to 2dgs normal and depth...')
        for i in trange(camera_num):
            viewpoint = self.scene[i]

            with torch.no_grad():
                render_pkg = self.renderImage(viewpoint)

                for key, value in render_pkg.items():
                    render_pkg[key] = value.detach()

            fc_normal = NVDiffRastRenderer.renderNormal(
                fc_mesh,
                viewpoint._cam,
                vertices_tensor=vertices,
            )['world']

            rend_normal  = render_pkg['rend_normal'].permute(1, 2, 0)

            fc_normal_loss = fc_normal_loss + l1_loss(fc_normal, rend_normal)

            fc_depth = NVDiffRastRenderer.renderDepth(
                fc_mesh,
                viewpoint._cam,
                bg_color=[0, 0, 0],
                vertices_tensor=vertices,
            )['depth']

            rend_depth = render_pkg['rend_depth']

            fc_depth_loss = fc_depth_loss + l1_loss(fc_depth, rend_depth)

            if L_dev is not None and L_dev.numel() > 0:
                dev_loss = dev_loss + L_dev.mean()

            '''
            faces = torch.from_numpy(fc_mesh.faces).long().to(self.device)
            if self.E_thinplate_base is None:
                with torch.no_grad():
                    V0 = torch.from_numpy(fc_mesh.vertices).float().to(self.device)
                    self.E_thinplate_base = thin_plate_energy(V0, faces)

                fc_mesh.export(self.save_result_folder_path + 'start_fc_mesh.ply')

            thinplate_loss = thinplate_loss + thin_plate_energy(vertices, faces, factor=self.E_thinplate_base)
            '''

        dev_loss = dev_loss / camera_num
        fc_normal_loss = fc_normal_loss / camera_num
        fc_depth_loss = fc_depth_loss / camera_num
        chamfer_loss = chamfer_loss / camera_num
        thinplate_loss = thinplate_loss / camera_num

        # loss
        total_loss = \
            lambda_dev * dev_loss + \
            lambda_fc_normal * fc_normal_loss + \
            lambda_fc_depth * fc_depth_loss + \
            lambda_chamfer * chamfer_loss + \
            lambda_thin_plate * thinplate_loss

        total_loss.backward()

        self.fc_optimizer.step()
        self.fc_optimizer.zero_grad()

        loss_dict = {
            'chamfer': chamfer_loss.item(),
            'dev': dev_loss.item(),
            'fc_normal': fc_normal_loss.item(),
            'fc_depth': fc_depth_loss.item(),
            'thinplate': thinplate_loss.item(),
            'fc_total': total_loss.item(),
        }
        return loss_dict

    @torch.no_grad
    def logFCStep(
        self,
        iteration: int,
        loss_dict: dict,
        render_image_num: int=5,
    ) -> bool:
        for key, value in loss_dict.items():
            self.logger.addScalar('FCLoss/' + key, value, iteration)

        torch.cuda.empty_cache()

        fc_mesh = self.extractMesh()[0]
        for idx in trange(render_image_num):
            viewpoint = self.scene[idx]

            if self.logger.isValid():
                fc_normal = NVDiffRastRenderer.renderNormal(
                    fc_mesh,
                    viewpoint._cam,
                )['rgb_world'].permute(2, 0, 1)
                self.logger.summary_writer.add_images("view_{}/fc_normal".format(viewpoint.image_name), fc_normal[None], global_step=iteration)

                fc_depth = NVDiffRastRenderer.renderDepth(
                    fc_mesh,
                    viewpoint._cam,
                )['rgb'].permute(2, 0, 1)
                self.logger.summary_writer.add_images("view_{}/fc_depth".format(viewpoint.image_name), fc_depth[None], global_step=iteration)
        return True

    @torch.no_grad()
    def recordGaussianState(
        self,
        iteration: int=0,
        record_num: int=-1,
    ) -> bool:
        if record_num == 0:
            return True

        fc_mesh = self.extractMesh()[0]

        print('[INFO][FCTrainer::recordGaussianState]')
        print('\t start record gaussian state...')

        if record_num < 0:
            record_num = len(self.scene)

        for idx in trange(record_num):
            viewpoint = self.scene[idx]
            render_pkg = self.renderImage(viewpoint)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            self.logger.summary_writer.add_images("view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

            try:
                depth = render_pkg["surf_depth"]
                norm = depth.max()
                depth = depth / norm
                depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                self.logger.summary_writer.add_images("view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
            except:
                pass

            try:
                rend_alpha = render_pkg['rend_alpha']
                rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                self.logger.summary_writer.add_images("view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                self.logger.summary_writer.add_images("view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                self.logger.summary_writer.add_images("view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                rend_dist = render_pkg["rend_dist"]
                rend_dist = colormap(rend_dist.cpu().numpy()[0])
                self.logger.summary_writer.add_images("view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
            except:
                pass

            self.logger.summary_writer.add_images("view_{}/GT".format(viewpoint.image_name), gt_image[None], global_step=iteration)

            fc_normal = NVDiffRastRenderer.renderNormal(
                fc_mesh,
                viewpoint._cam,
            )['rgb_world'].permute(2, 0, 1)
            self.logger.summary_writer.add_images("view_{}/fc_normal".format(viewpoint.image_name), fc_normal[None], global_step=iteration)

            fc_depth = NVDiffRastRenderer.renderDepth(
                fc_mesh,
                viewpoint._cam,
            )['rgb'].permute(2, 0, 1)
            self.logger.summary_writer.add_images("view_{}/fc_depth".format(viewpoint.image_name), fc_depth[None], global_step=iteration)
        return True

    @torch.no_grad()
    def recordGrads(self, render_pkg: dict) -> bool:
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        return True

    @torch.no_grad()
    def densifyStep(self, render_pkg: dict) -> bool:
        size_threshold = 20
        my_viewpoint_stack = self.scene.train_cameras
        camlist = sampling_cameras(my_viewpoint_stack)

        # The multiview consistent densification of fastgs
        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, self.gaussians, self.pipe, self.background, self.opt, DENSIFY=True)
        self.gaussians.densify_and_prune_fastgs(
            max_screen_size = size_threshold,
            min_opacity = 0.005,
            extent = self.scene.cameras_extent,
            radii=render_pkg['radii'],
            args = self.opt,
            importance_score = importance_score,
            pruning_score = pruning_score,
        )
        return True

    @torch.no_grad()
    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
        return True

    @torch.no_grad()
    def resetScaling(self) -> bool:
        self.gaussians.reset_scaling()
        return True

    @torch.no_grad()
    def pruneGaussiansOutsideMasks(self) -> bool:
        """删除不在任一 mask 内的 gaussian：将每个 gaussian 投影到所有带 mask 的视角，若在任一视角的 mask 内则保留，否则删除。"""
        cams_with_mask = [c for c in self.scene.train_cameras if getattr(c, 'gt_alpha_mask', None) is not None]
        if not cams_with_mask:
            return True

        xyz = self.gaussians.get_xyz  # (N, 3) on cuda
        N = xyz.shape[0]
        if N == 0:
            return True

        device = xyz.device
        inside_any = torch.zeros(N, dtype=torch.bool, device=device)

        for cam in cams_with_mask:
            # full_proj_transform: (4, 4), world point -> NDC
            proj = cam.full_proj_transform.to(device)
            mask = cam.gt_alpha_mask.to(device)  # (1, H, W)
            H, W = mask.shape[1], mask.shape[2]

            xyz_h = torch.cat([xyz, torch.ones(N, 1, device=device, dtype=xyz.dtype)], dim=1)  # (N, 4)
            proj_pts = (proj.unsqueeze(0) @ xyz_h.unsqueeze(-1)).squeeze(-1)  # (N, 4)
            w = proj_pts[:, 3].clamp(min=1e-6)
            ndc_x = proj_pts[:, 0] / w
            ndc_y = proj_pts[:, 1] / w
            ndc_z = proj_pts[:, 2] / w

            # 在相机前方 (NDC z 通常在 [0,1] 或 [-1,1] 视实现而定，这里取可见为 z > 0)
            in_front = ndc_z > 0
            # NDC -> pixel: x right, y down; NDC x,y in [-1,1]
            u = ((ndc_x + 1.0) * 0.5 * (W - 1)).long().clamp(0, W - 1)
            v = ((1.0 - ndc_y) * 0.5 * (H - 1)).long().clamp(0, H - 1)
            mask_val = mask[0, v, u]  # (N,)
            inside_mask = (mask_val > 0.5) & in_front
            inside_any = inside_any | inside_mask

        to_remove = ~inside_any
        n_remove = to_remove.sum().item()
        if n_remove > 0:
            self.gaussians.prune_points(to_remove)
            if self.surface_confidence is not None and self.surface_confidence.shape[0] == N:
                self.surface_confidence = self.surface_confidence[~to_remove]
        return True

    @torch.no_grad()
    def finalPrune(self) -> bool:
        my_viewpoint_stack = self.scene.train_cameras
        camlist = sampling_cameras(my_viewpoint_stack)

        _, pruning_score = compute_gaussian_score_fastgs(camlist, self.gaussians, self.pipe, self.background, self.opt)
        self.gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
        return True

    @torch.no_grad()
    def updateGSParams(self, iteration: int) -> bool:
        self.gaussians.optimizer_step(iteration)
        return True

    @torch.no_grad()
    def saveScene(self, iteration: int) -> bool:
        point_cloud_path = os.path.join(self.dataset.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        fc_mesh = self.extractMesh()[0]
        fc_mesh.export(os.path.join(point_cloud_path, 'fc_mesh.ply'))
        return True

    def train(self, iteration_num: int = 30000):
        iteration = 0

        self.recordGaussianState(iteration, record_num=1)

        while iteration < iteration_num:
            print('[INFO][FCTrainer::train]')
            print('\t start train gs...')
            for i in trange(len(self.scene)):
                iteration += 1

                viewpoint_cam = self.scene[i]

                render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

                self.logStep(
                    iteration,
                    loss_dict,
                )

                if iteration >= self.log_start_iter:
                    if iteration % self.test_freq == 0:
                        self.logImageStep(
                            iteration,
                            render_image_num=1,
                            is_fast=True,
                        )

                    if iteration % self.save_freq == 0:
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        self.saveScene(iteration)

                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.recordGrads(render_pkg)
                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        self.densifyStep(render_pkg)

                    if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.resetOpacity()

                    if iteration % self.opt.scaling_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.resetScaling()

                # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
                # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
                # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
                if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                    self.finalPrune()

                # 每个 step 删除不在任一 mask 内的 gaussian
                # self.pruneGaussiansOutsideMasks()

                self.updateGSParams(iteration)

                self.iteration = iteration

            if iteration >= self.log_start_iter:
                print('[INFO][FCTrainer::train]')
                print('\t start train fc...')
                loss_dict = self.trainFCStep()
                self.logFCStep(
                    iteration,
                    loss_dict,
                    render_image_num=1,
                )
        return True

    def exportMesh(
        self,
        mesh_res: int = 1024,
        voxel_size: float = -1.0,
        depth_trunc: float = -1.0,
        sdf_trunc: float = -1.0,
        num_cluster: int = 50,
    ) -> bool:
        export_scene = deepcopy(self.scene)
        export_gaussians = export_scene.gaussians

        train_dir = os.path.join(self.dataset.model_path, 'mesh', 'iter_' + str(self.iteration))
        gaussExtractor = GaussianExtractor(export_gaussians, render, self.pipe, bg_color=self.background.cpu().numpy())

        print('[INFO][Trainer::exportMesh]')
        print("\t start export mesh...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(export_scene.train_cameras)

        # extract the mesh and save
        name = 'fuse.ply'
        depth_trunc = (gaussExtractor.radius * 2.0) if depth_trunc < 0  else depth_trunc
        voxel_size = (depth_trunc / mesh_res) if voxel_size < 0 else voxel_size
        sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
        return True
