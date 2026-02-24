import os
import sys
import torch
import open3d as o3d

from torch import nn
from tqdm import trange
from typing import Tuple
from copy import deepcopy
from argparse import ArgumentParser

from fused_ssim import fused_ssim

from utils.mesh_utils import GaussianExtractor, post_process_mesh

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Method.path import createFileFolder
from base_gs_trainer.Method.general_utils import colormap
from base_gs_trainer.Loss.chamfer import chamferLossFn
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from flexi_cubes.Module.fc_convertor import FCConvertor

from mv_fc_recon.Loss.mesh_geo_energy import thin_plate_energy

from twod_gs.Config.config import ModelParams, PipelineParams, OptimizationParams
from twod_gs.Method.render_kernel import render
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
        lambda_opacity: float = 1e-6,
        lambda_scaling: float = 1.0,
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
            normal_loss = normal_error.mean()

        lambda_dist = lambda_dist if iteration > 1000 else 0.0
        dist_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            valid_dist_idxs = torch.where(rend_dist != 0)
            valid_rend_dist = rend_dist[valid_dist_idxs]
            dist_loss = valid_rend_dist.mean()

        # Phase A: Surface selection — push non-surface to 0 (opacity, scale)；Phase B 开始后彻底关闭
        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0:
            opacity_loss = nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0:
            scaling_loss = nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # loss
        total_loss = \
            rgb_loss + \
            lambda_dist * dist_loss + \
            lambda_normal * normal_loss + \
            lambda_opacity * opacity_loss + \
            lambda_scaling * scaling_loss

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
        camera_batch_size: int = 4,
    ) -> dict:
        camera_num = len(self.scene)
        camera_batch_size = min(camera_batch_size, camera_num)
        num_batches = (camera_num + camera_batch_size - 1) // camera_batch_size

        # 用于记录标量 loss（仅日志），不参与反向
        sum_fc_normal = 0.0
        sum_fc_depth = 0.0
        sum_dev = 0.0
        sum_chamfer = 0.0
        sum_thinplate = 0.0

        self.fc_optimizer.zero_grad()

        print('[INFO][FCTrainer::trainFCStep]')
        print('\t start match fc to 2dgs normal and depth (batch_size={})...'.format(camera_batch_size))

        for start in trange(0, camera_num, camera_batch_size):
            end = min(start + camera_batch_size, camera_num)
            batch_size = end - start

            # 每个 batch 重新 extractMesh，保证本 batch 的 backward 有完整计算图
            fc_mesh, vertices, L_dev = self.extractMesh()
            if L_dev is not None and L_dev.numel() > 0:
                sum_dev += L_dev.mean().item() * batch_size

            batch_fc_normal_loss = torch.tensor(0.0, device=self.device)
            batch_fc_depth_loss = torch.tensor(0.0, device=self.device)

            for i in range(start, end):
                viewpoint = self.scene[i]

                with torch.no_grad():
                    render_pkg = self.renderImage(viewpoint)
                    for key in list(render_pkg.keys()):
                        render_pkg[key] = render_pkg[key].detach()

                fc_normal = NVDiffRastRenderer.renderNormal(
                    fc_mesh,
                    viewpoint._cam,
                    vertices_tensor=vertices,
                )['world']
                rend_normal = render_pkg['rend_normal'].permute(1, 2, 0)
                batch_fc_normal_loss = batch_fc_normal_loss + l1_loss(fc_normal, rend_normal)

                fc_depth = NVDiffRastRenderer.renderDepth(
                    fc_mesh,
                    viewpoint._cam,
                    bg_color=[0, 0, 0],
                    vertices_tensor=vertices,
                )['depth']
                rend_depth = render_pkg['rend_depth']
                batch_fc_depth_loss = batch_fc_depth_loss + l1_loss(fc_depth, rend_depth)

            batch_fc_normal_loss = batch_fc_normal_loss / batch_size
            batch_fc_depth_loss = batch_fc_depth_loss / batch_size
            sum_fc_normal += batch_fc_normal_loss.item() * batch_size
            sum_fc_depth += batch_fc_depth_loss.item() * batch_size

            # 本 batch 的加权 loss，梯度累积后等价于全图平均
            scale = batch_size / camera_num
            batch_total = (
                lambda_fc_normal * batch_fc_normal_loss + lambda_fc_depth * batch_fc_depth_loss
            ) * scale
            if L_dev is not None and L_dev.numel() > 0:
                batch_total = batch_total + lambda_dev * L_dev.mean() / num_batches
            batch_total.backward()

            del fc_mesh, vertices, L_dev, batch_fc_normal_loss, batch_fc_depth_loss, batch_total
            torch.cuda.empty_cache()

        self.fc_optimizer.step()

        # 汇总标量 loss（与原先语义一致：全图平均）
        fc_normal_loss = sum_fc_normal / camera_num
        fc_depth_loss = sum_fc_depth / camera_num
        dev_loss = sum_dev / camera_num if camera_num else 0.0
        chamfer_loss = sum_chamfer / camera_num
        thinplate_loss = sum_thinplate / camera_num
        total_loss_scalar = (
            lambda_dev * dev_loss
            + lambda_fc_normal * fc_normal_loss
            + lambda_fc_depth * fc_depth_loss
            + lambda_chamfer * chamfer_loss
            + lambda_thin_plate * thinplate_loss
        )

        loss_dict = {
            'chamfer': chamfer_loss,
            'dev': dev_loss,
            'fc_normal': fc_normal_loss,
            'fc_depth': fc_depth_loss,
            'thinplate': thinplate_loss,
            'fc_total': total_loss_scalar,
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
    def densifyStep(self) -> bool:
        size_threshold = 20
        self.gaussians.densify_and_prune(
            self.opt.densify_grad_threshold,
            self.opt.opacity_cull,
            self.scene.cameras_extent,
            size_threshold,
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
    def updateGSParams(self) -> bool:
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        return True

    @torch.no_grad()
    def saveGaussians(self, iteration: int) -> bool:
        save_gaussians_file_path = os.path.join(self.dataset.model_path, f"point_cloud/{iteration:06d}.ply")
        createFileFolder(save_gaussians_file_path)
        self.gaussians.save_ply(save_gaussians_file_path)
        return True

    @torch.no_grad()
    def saveFC(self, iteration: int) -> bool:
        save_fc_mesh_file_path = os.path.join(self.dataset.model_path, f"fc_mesh/{iteration:06d}.ply")
        createFileFolder(save_fc_mesh_file_path)
        fc_mesh = self.extractMesh()[0]
        fc_mesh.export(save_fc_mesh_file_path)
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
                        self.saveGaussians(iteration)

                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.recordGrads(render_pkg)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        self.densifyStep()

                    if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.resetOpacity()

                    if iteration % self.opt.scaling_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.resetScaling()

                self.updateGSParams()

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
                self.saveFC(iteration)
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
