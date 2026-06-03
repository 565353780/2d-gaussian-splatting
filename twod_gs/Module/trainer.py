import os
import torch
import open3d as o3d

from torch import nn
from tqdm import tqdm
from typing import Tuple
from copy import deepcopy

from fused_ssim import fused_ssim

from twod_gs.Method.mesh_utils import GaussianExtractor, post_process_mesh

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from twod_gs.Config.config import ModelParams, PipelineParams, OptimizationParams
from twod_gs.Method.render_kernel import render
from twod_gs.Model.gs import GaussianModel


class Trainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
    ) -> None:
        self.dataset = ModelParams.default()
        self.opt = OptimizationParams.default()
        self.pipe = PipelineParams.default()

        self.dataset.source_path = os.path.abspath(colmap_data_folder_path)
        self.dataset.model_path = save_result_folder_path

        print("Optimizing " + self.dataset.model_path)

        self.gaussians = GaussianModel(self.dataset.sh_degree)

        # 训练阶段边界：
        self.final_train_iter = 12_000
        # 屏幕空间梯度范数大于该阈值才视为本轮“收到过梯度”（即在渲染中可见）
        self.invisible_grad_eps = 1e-12
        # epoch 级可见性统计作为 per-point 跟随张量由 GaussianModel 维护（见 gs.py），
        # 随 densify/prune 同步增删索引，可与致密化全程并行

        BaseGSTrainer.__init__(
            self,
            colmap_data_folder_path=colmap_data_folder_path,
            device=device,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            test_freq=test_freq,
            save_freq=save_freq,
        )
        return

    def renderImage(self, viewpoint_cam) -> dict:
        return render(viewpoint_cam, self.gaussians, self.pipe, self.background)

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

        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0:
            cur_opacity = self.gaussians.get_opacity
            target_opacity = torch.full_like(cur_opacity, 1.0)
            opacity_loss = nn.MSELoss()(cur_opacity, target_opacity)

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0:
            scaling_loss = nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # loss
        total_loss = rgb_loss + \
            lambda_dist * dist_loss + \
            lambda_normal * normal_loss + \
            lambda_opacity * opacity_loss + \
            lambda_scaling * scaling_loss

        total_loss.backward()

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

    @torch.no_grad()
    def recordGrads(self, render_pkg: dict) -> bool:
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        return True

    @torch.no_grad()
    def resetEpochVisibility(self) -> bool:
        # 可见性统计作为 per-point 跟随张量由 GaussianModel 维护，随 densify/prune 同步索引
        self.gaussians.reset_epoch_visibility()
        return True

    @torch.no_grad()
    def recordEpochVisibility(self, render_pkg: dict) -> bool:
        # 用屏幕空间 means2D 的梯度作为“本步是否对图像有贡献”的依据；
        # 不能用 _opacity.grad，因为 opacity 正则会让所有 GS 都有参数梯度。
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        grad = viewspace_point_tensor.grad
        if grad is None:
            return True

        # received_grad 基于本步渲染，长度恒等于当前 GS 数量；GaussianModel 内的
        # 统计张量已随 densify/prune 同步索引，二者天然对齐。
        grad_norm = torch.norm(grad, dim=-1)
        received_grad = visibility_filter & (grad_norm > self.invisible_grad_eps)
        self.gaussians.accumulate_epoch_visibility(received_grad)
        return True

    @torch.no_grad()
    def pruneInvisibleGS(self) -> int:
        # 剔除整轮相机都没有收到屏幕空间梯度的 GS（在渲染中不可见的无效 GS）
        if not self.gaussians.epoch_visibility_enabled:
            return 0

        prune_mask = ~self.gaussians.epoch_has_image_grad
        prune_num = int(prune_mask.sum().item())
        if prune_num > 0:
            self.gaussians.prune_points(prune_mask)

        self.resetEpochVisibility()
        return prune_num

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
    def saveScene(self, iteration: int) -> bool:
        point_cloud_path = os.path.join(self.dataset.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        return True

    def train(self, iteration_num: int = 12000):
        # 1.2w 后 normal loss 开始震荡，硬上限到 final_train_iter；更小的步数用于快速实验
        # 记录外部请求的步数，最终结果统一以该步数命名保存，
        # 与 fast_gs 的 ``point_cloud/iteration_{steps}`` 保存/读取方式对齐
        requested_iteration_num = iteration_num
        iteration_num = min(iteration_num, self.final_train_iter)

        epoch_len = max(1, len(self.scene))
        last_prune_num = 0

        progress_bar = tqdm(desc="Training progress", total=iteration_num)
        iteration = 0
        for _ in range(iteration_num):
            iteration += 1

            viewpoint_cam = self.scene[iteration]

            render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

            # backward 已在 trainStep 内完成，此处趁梯度未清零记录本步屏幕空间可见性。
            # 可见性统计是 per-point 跟随张量，densify/prune 改变点数时索引会同步维护，
            # 因此可与致密化全程并行、随时触发。
            self.recordEpochVisibility(render_pkg)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "distort": f"{loss_dict['dist']:.{5}f}",
                    "normal": f"{loss_dict['normal']:.{5}f}",
                    "pruned": f"{last_prune_num}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.logStep(iteration, loss_dict)

            if iteration % self.test_freq == 0:
                self.logImageStep(
                    iteration,
                    render_image_num=1,
                    is_fast=True,
                )

            if iteration % self.save_freq == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.saveScene(iteration)

            # densify 在 densify_until_iter 后停止；可见性统计与剪枝已能与 densify 并行，
            # 二者无需互斥
            if iteration < self.opt.densify_until_iter:
                self.recordGrads(render_pkg)

                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    self.densifyStep()

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetOpacity()

                if iteration % self.opt.scaling_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetScaling()

            self.updateGSParams()

            # 每过完一整轮相机，剔除整轮都没收到屏幕空间梯度的不可见 GS
            if iteration % epoch_len == 0:
                last_prune_num = self.pruneInvisibleGS()
                if last_prune_num > 0:
                    print("\n[ITER {}] Pruned {} invisible gaussians, remaining {}".format(
                        iteration, last_prune_num, len(self.gaussians.get_xyz)))

            self.iteration = iteration

        # 训练结束后统一以外部请求的步数命名保存最终结果，
        # 保证 ``point_cloud/iteration_{steps}/point_cloud.ply`` 始终存在
        print("\n[ITER {}] Saving Gaussians".format(requested_iteration_num))
        self.saveScene(requested_iteration_num)
        self.iteration = requested_iteration_num
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
