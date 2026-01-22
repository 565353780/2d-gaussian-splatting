import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from random import randint

from mash_2dgs.Config.joint_optimization_params import JointOptimizationParams
from mash_2dgs.Module.trainer import Trainer
from mash_2dgs.Module.mash_refiner import MashRefiner

class JointTrainer(object):
    def __init__(self,
                 source_path: str,
                 images: str,
                 save_result_folder_path: str,
                 ply_file_path: Union[str, None]=None,
                 anchor_cluster_point_num: int=100,
                 ) -> None:
        self.anchor_cluster_point_num = anchor_cluster_point_num

        self.trainer = Trainer(source_path, images, save_result_folder_path, ply_file_path, JointOptimizationParams)

        anchor_num = 400
        mask_degree_max = 3
        sh_degree_max = 2
        mask_boundary_sample_num = 90
        sample_polar_num = 1000
        sample_point_scale = 0.8
        self.mash_refiner = MashRefiner(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            mask_boundary_sample_num,
            sample_polar_num,
            sample_point_scale,
        )

        self.save_freq = 5000

        self.viewpoint_stack = None

        self.iteration = 1
        self.last_save_iteration = None
        self.last_save_gs_ply_file_path = None
        self.last_save_refined_surface_pcd_file_path = None
        self.surface_points = None
        return

    def getImage(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        return viewpoint_cam

    def trainStep(self,
                  lambda_dssim: float = 0.2,
                  lambda_normal: float = 0.01,
                  lambda_dist: float = 100000.0,
                  lambda_opacity: float = 0.001,
                  lambda_scaling: float = 0.001,
                  lambda_surface: float = 0.001,
                  maximum_opacity: bool = False,
                  ):
        return self.trainer.trainStepWithSuperParams(
            self.iteration,
            self.getImage(),
            lambda_dssim,
            lambda_normal,
            lambda_dist,
            lambda_opacity,
            lambda_scaling,
            lambda_surface,
            maximum_opacity,
            self.surface_points,
        )

    def logStep(self, loss_dict: dict) -> bool:
        self.trainer.logStep(self.iteration, loss_dict)
        return True

    def updateProgressBar(self, progress_bar, loss_dict: dict) -> bool:
        if self.iteration % 10 != 0:
            return True

        bar_loss_dict = {
            "rgb": f"{loss_dict['rgb']:.{5}f}",
            "distort": f"{loss_dict['dist']:.{5}f}",
            "normal": f"{loss_dict['normal']:.{5}f}",
            "Points": f"{len(self.trainer.gaussians.get_xyz)}"
        }
        progress_bar.set_postfix(bar_loss_dict)
        progress_bar.update(10)
        return True

    def saveScene(self, force: bool = False) -> bool:
        if self.iteration % self.save_freq != 0:
            if not force:
                return True

        if self.iteration == self.last_save_iteration:
            return True

        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
        self.last_save_gs_ply_file_path = self.trainer.saveScene(self.iteration)
        return True

    def densifyStep(self, render_pkg) -> bool:
        if self.iteration < self.trainer.opt.densify_until_iter or self.trainer.opt.densify_until_iter < 0:
            self.trainer.recordGrads(render_pkg)
            if self.iteration > self.trainer.opt.densify_from_iter and self.iteration % self.trainer.opt.densification_interval == 0:
                self.trainer.densifyStep()
        return True

    def resetOpacity(self) -> bool:
        if self.iteration % self.trainer.opt.opacity_reset_interval == 0 or (self.trainer.dataset.white_background and self.iteration == self.trainer.opt.densify_from_iter):
            self.trainer.resetOpacity()
        return True

    def preProcessGS(self, progress_bar, loss_dict: dict) -> bool:
        self.updateProgressBar(progress_bar, loss_dict)
        self.logStep(loss_dict)
        self.saveScene()
        return True

    def postProcessGS(self, loss_dict: dict) -> bool:
        self.trainer.updateGSParams()
        self.trainer.renderForViewer(loss_dict)
        return True

    def refineWithMash(self) -> bool:
        if not hasattr(self.trainer.opt, 'mash_refine_interval'):
            print('[ERROR][JointTrainer::refineWithMash]')
            print('\t mash_refine_interval not found in trainer.opt!')
            return False

        if self.iteration % self.trainer.opt.mash_refine_interval != 0:
            return True

        self.saveScene(True)

        if self.last_save_gs_ply_file_path is None:
            print('[ERROR][JointTrainer::refineWithMash]')
            print('\t last save gs ply file not exist!')
            return False

        save_refined_pcd_file_path = self.last_save_gs_ply_file_path.replace('/point_cloud.ply', '/point_cloud_refined.ply')

        gs_pcd = o3d.io.read_point_cloud(self.last_save_gs_ply_file_path)
        gs_points_array = np.asarray(gs_pcd.points)
        gs_points = torch.from_numpy(gs_points_array)

        self.mash_refiner.anchor_num = int(gs_points.shape[0] / self.anchor_cluster_point_num)
        print('[INFO][JointTrainer::refineWithMash]')
        print('\t generated anchor num :', self.mash_refiner.anchor_num)

        self.surface_points = self.mash_refiner.toSurfacePoints(gs_points).detach().clone().unsqueeze(0)
        print('[INFO][JointTrainer::refineWithMash]')
        print('\t refined surface points size :', self.surface_points.shape)

        self.mash_refiner.saveSurfacePointsAsPcdFile(save_refined_pcd_file_path, True)

        if os.path.exists(save_refined_pcd_file_path):
            self.last_save_refined_surface_pcd_file_path = save_refined_pcd_file_path
        return True

    def trainForever(self) -> bool:
        progress_bar = tqdm(desc="MASH-2DGS forever training progress")

        while True:
            render_pkg, loss_dict = self.trainStep()

            self.preProcessGS(progress_bar, loss_dict)

            self.densifyStep(render_pkg)

            self.resetOpacity()

            self.postProcessGS(loss_dict)

            self.refineWithMash()

            self.iteration += 1

    def train(self, iteration_num: int = -1) -> bool:
        if iteration_num < 0:
            return self.trainForever()

        lambda_dssim = 0.2
        lambda_normal = 0.01
        lambda_dist = 100000.0
        lambda_opacity = 0.01
        lambda_scaling = 0.1
        lambda_surface = 0.001
        maximum_opacity = False

        progress_bar = tqdm(desc="MASH-2DGS training progress", total=iteration_num)

        for i in range(iteration_num):
            if self.iteration < 20000 and False:
                new_lambda_surface = 0.0
            else:
                new_lambda_surface = lambda_surface

            render_pkg, loss_dict = self.trainStep(
                lambda_dssim,
                lambda_normal,
                lambda_dist,
                lambda_opacity,
                lambda_scaling,
                new_lambda_surface,
                maximum_opacity,
            )

            self.preProcessGS(progress_bar, loss_dict)

            self.densifyStep(render_pkg)

            self.resetOpacity()

            self.postProcessGS(loss_dict)

            if new_lambda_surface > 0:
                if i < iteration_num - 1:
                    self.refineWithMash()

            self.iteration += 1

        return True

    def convertToMesh(self, conda_env_name: str, iteration: int) -> bool:
        return self.trainer.convertToMesh(conda_env_name, iteration)
