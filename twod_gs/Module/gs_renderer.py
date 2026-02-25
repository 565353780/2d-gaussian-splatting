import os
import sys
import torch

from tqdm import tqdm
from typing import List
from argparse import ArgumentParser

from base_gs_trainer.Data.gs_camera import GSCamera

from camera_control.Module.camera import Camera

from twod_gs.Config.config import PipelineParams
from twod_gs.Model.gs import GaussianModel
from twod_gs.Method.render_kernel import render


class GSRenderer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def renderCameras(
        gs_ply_file_path: str,
        camera_list: List[Camera],
        sh_degree: int = 3,
        bg_color: list=[1, 1, 1],
        device: str='cuda:0',
    ) -> List:
        if len(camera_list) == 0:
            return []

        if not os.path.exists(gs_ply_file_path):
            print('[ERROR][GSRenderer::renderCameras]')
            print('\t gs ply file not exist!')
            print('\t gs_ply_file_path:', gs_ply_file_path)
            return []

        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        pp = PipelineParams(parser)
        args = parser.parse_args(sys.argv[1:])

        pipe = pp.extract(args)

        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        gaussians = GaussianModel(sh_degree=sh_degree)
        gaussians.load_ply(gs_ply_file_path)

        print('[INFO][GSRenderer::renderCameras]')
        print('\t start render cameras...')
        render_list = []
        with torch.no_grad():
            for camera in tqdm(camera_list):
                gs_camera = GSCamera(camera, device)
                render_dict = render(gs_camera, gaussians, pipe, background)
                # 立即转移到 CPU 以释放显存，且不参与梯度计算
                for k, v in render_dict.items():
                    if isinstance(v, torch.Tensor):
                        render_dict[k] = v.cpu()
                render_list.append(render_dict)
        return render_list
