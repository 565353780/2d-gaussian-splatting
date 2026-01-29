import os
import sys
import torch
import open3d as o3d

from torch import nn
from tqdm import tqdm
from typing import Tuple
from copy import deepcopy

from lpipsPyTorch import lpips
from fused_ssim import fused_ssim as fast_ssim

from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

from base_trainer.Module.logger import Logger
from base_trainer.Module.base_trainer import BaseTrainer

from twod_gs.Loss.l1 import l1_loss
from twod_gs.Metric.psnr import psnr
from twod_gs.Method.fast_utils import sampling_cameras, compute_gaussian_score_fastgs
from twod_gs.Dataset.scene import Scene
from twod_gs.Model.gs import GaussianModel


class Trainer(object):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
    ) -> None:
        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.test_freq = 500
        self.save_freq = 500

        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(sys.argv[1:])

        args.source_path = colmap_data_folder_path
        args.images = 'masked_images'
        args.white_background = True
        args.resolution = 2
        args.model_path = save_result_folder_path

        print("Optimizing " + args.model_path)

        # Initialize system state (RNG)
        safe_state(silent=False)

        torch.autograd.set_detect_anomaly(False)

        os.makedirs(args.model_path, exist_ok=True)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

        self.logger = Logger()

        BaseTrainer.initRecords(self)
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
        lambda_opacity: float = 0.01,
        lambda_scaling: float = 0.01,
        maximum_opacity: bool = False,
    ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        render_pkg = self.renderImage(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
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
        #FIXME: seems this dist not work
        dist_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            valid_dist_idxs = torch.where(rend_dist != 0)
            valid_rend_dist = rend_dist[valid_dist_idxs]
            dist_loss = lambda_dist * valid_rend_dist.mean()

        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0:
            if maximum_opacity:
                opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.ones_like(self.gaussians._opacity))
            else:
                opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0:
            scaling_loss = lambda_scaling * nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # loss
        total_loss = rgb_loss + dist_loss + normal_loss + opacity_loss + scaling_loss

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

    @torch.no_grad
    def logStep(self, iteration: int, loss_dict: dict) -> bool:
        reg_loss = loss_dict['reg']
        ssim_loss = loss_dict['ssim']
        rgb_loss = loss_dict['rgb']
        dist_loss = loss_dict['dist']
        normal_loss = loss_dict['normal']
        opacity_loss = loss_dict['opacity']
        scaling_loss = loss_dict['scaling']
        total_loss = loss_dict['total']

        # Log and save
        self.logger.addScalar('Loss/reg', reg_loss, iteration)
        self.logger.addScalar('Loss/ssim', ssim_loss, iteration)
        self.logger.addScalar('Loss/rgb', rgb_loss, iteration)
        self.logger.addScalar('Loss/dist', dist_loss, iteration)
        self.logger.addScalar('Loss/normal', normal_loss, iteration)
        self.logger.addScalar('Loss/opacity', opacity_loss, iteration)
        self.logger.addScalar('Loss/scaling', scaling_loss, iteration)
        self.logger.addScalar('Loss/total', total_loss, iteration)

        self.logger.addScalar('Gaussian/total_points', self.gaussians.get_xyz.shape[0], iteration)
        self.logger.addScalar('Gaussian/scale', torch.mean(self.gaussians.get_scaling).detach().clone().cpu().numpy(), iteration)
        self.logger.addScalar('Gaussian/opacity', torch.mean(self.gaussians.get_opacity).detach().clone().cpu().numpy(), iteration)

        # Report test and samples of training set
        if iteration % self.test_freq == 0:
            torch.cuda.empty_cache()
            config = {'name': 'train', 'cameras' : [self.scene[idx % len(self.scene)] for idx in range(5, 30, 5)]}
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = self.renderImage(viewpoint)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if self.logger.isValid() and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        self.logger.summary_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        self.logger.summary_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == 1:
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                self.logger.addScalar('Eval/l1', l1_test, iteration)
                self.logger.addScalar('Eval/psnr', psnr_test, iteration)
                self.logger.addScalar('Eval/ssim', ssim_test, iteration)
                self.logger.addScalar('Eval/lpips', lpips_test, iteration)

            torch.cuda.empty_cache()
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
        return True

    def train(self, iteration_num: int = 30000):
        progress_bar = tqdm(desc="Training progress", total=iteration_num)
        iteration = 1
        for _ in range(iteration_num):
            viewpoint_cam = self.scene[iteration]

            render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "distort": f"{loss_dict['dist']:.{5}f}",
                    "normal": f"{loss_dict['normal']:.{5}f}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.logStep(iteration, loss_dict)

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

            self.updateGSParams(iteration)

            iteration += 1
            self.iteration = iteration
        return True

    def exportMesh(
        self,
        mesh_res: int = 1024,
        voxel_size: float = -1.0,
        depth_trunc: float = -1.0,
        sdf_trunc: float = -1.0,
        num_cluster: int = 1,
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
