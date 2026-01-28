import os
import sys
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

from base_trainer.Module.logger import Logger
from base_trainer.Module.base_trainer import BaseTrainer

from twod_gs.Loss.l1 import l1_loss
from twod_gs.Loss.ssim import ssim
from twod_gs.Metric.psnr import psnr
from twod_gs.Dataset.scene import Scene
from twod_gs.Method.cmd import runCMD
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
        lambda_dist: float = 0.0,
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
        ssim_loss = 1.0 - ssim(image, gt_image)
        rgb_loss = (1.0 - lambda_dssim) * reg_loss + lambda_dssim * ssim_loss

        lambda_normal = lambda_normal if iteration > 7000 else 0.0
        normal_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_normal > 0:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_dot = torch.sum(rend_normal * surf_normal, dim=0)

            valid_dot_idxs = torch.where(normal_dot != 0)
            valid_normal_dot = normal_dot[valid_dot_idxs]

            normal_error = (1 - valid_normal_dot)
            normal_loss = lambda_normal * normal_error.mean()

        lambda_dist = lambda_dist if iteration > 3000 else 0.0
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
                psnr_test = 0.0
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

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                self.logger.addScalar('Val/l1', l1_test, iteration)
                self.logger.addScalar('Val/psnr', psnr_test, iteration)

            torch.cuda.empty_cache()
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
        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.opacity_cull, self.scene.cameras_extent, size_threshold)
        return True

    @torch.no_grad()
    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
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
                    self.densifyStep(iteration)

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetOpacity()

            self.updateGSParams()

            iteration += 1
        return True

    def convertToMesh(self, conda_env_name: str, iteration: int) -> bool:
        saved_result_folder_path = self.save_result_folder_path + 'point_cloud/iteration_' + str(iteration) + '/'
        if not os.path.exists(saved_result_folder_path):
            print('[ERROR][Trainer::convertToMesh]')
            print('\t saved result not found!')
            print('\t saved_result_folder_path :', saved_result_folder_path)
            return False

        command = 'conda run -n ' + conda_env_name + ' python render.py' + \
            ' -s ' + self.dataset.source_path + \
            ' -m ' + self.save_result_folder_path + \
            ' --num_cluster 1' + \
            ' --iteration ' + str(iteration)

        os.system(command)
        return True

        if not runCMD(command):
            print('[ERROR][Trainer::convertToMesh]')
            print('\t runCMD failed!')
            print('\t command :', command)
            return False

        return True
