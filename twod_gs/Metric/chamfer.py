import os
import torch
import numpy as np
import open3d as o3d
from typing import Union, overload

import mash_cpp

from mash_2dgs.Config.constant import EPSILON
from mash_2dgs.Method.path import removeFile

def toChamferDistance(points_1: Union[torch.Tensor, np.ndarray], points_2: Union[torch.Tensor, np.ndarray], device: str = 'cuda') -> float:
    if isinstance(points_1, np.ndarray):
        points_1 = torch.from_numpy(points_1)
    if isinstance(points_2, np.ndarray):
        points_2 = torch.from_numpy(points_2)

    points_1 = points_1.type(torch.float32).to(device)
    points_2 = points_2.type(torch.float32).to(device)

    if len(points_1.shape) < 3:
        points_1 = points_1.unsqueeze(0)
    if len(points_2.shape) < 3:
        points_2 = points_2.unsqueeze(0)

    fit_dists2, coverage_dists2, _, _ = mash_cpp.toChamferDistance(points_1, points_2)

    fit_dists2 = fit_dists2.squeeze(0)
    coverage_dists2 = coverage_dists2.squeeze(0)

    fit_dists = torch.sqrt(fit_dists2 + EPSILON)
    coverage_dists = torch.sqrt(coverage_dists2 + EPSILON)

    fit_loss = torch.mean(fit_dists)
    coverage_loss = torch.mean(coverage_dists)

    chamfer_distance = fit_loss + coverage_loss

    return chamfer_distance.item()

def toChamferDistanceFromPcdFile(pcd_file_path_1: str, pcd_file_path_2: str) -> float:
    if not os.path.exists(pcd_file_path_1):
        print('[ERROR][chamfer::toChamferDistanceFromPcdFile]')
        print('\t pcd file not found!')
        print('\t pcd_file_path_1 :', pcd_file_path_1)
        return -1.0

    if not os.path.exists(pcd_file_path_2):
        print('[ERROR][chamfer::toChamferDistanceFromPcdFile]')
        print('\t pcd file not found!')
        print('\t pcd_file_path_2 :', pcd_file_path_2)
        return -1.0

    pcd_1 = o3d.io.read_point_cloud(pcd_file_path_1)
    pcd_2 = o3d.io.read_point_cloud(pcd_file_path_2)

    points_1 = np.asarray(pcd_1.points)
    points_2 = np.asarray(pcd_2.points)

    return toChamferDistance(points_1, points_2)

def toChamferDistanceFromMeshFile(mesh_file_path_1: str, mesh_file_path_2: str, sample_point_num: int, overwrite: bool = False) -> float:
    if not os.path.exists(mesh_file_path_1):
        print('[ERROR][chamfer::toChamferDistanceFromMeshFile]')
        print('\t mesh file not found!')
        print('\t mesh_file_path_1 :', mesh_file_path_1)
        return -1.0

    if not os.path.exists(mesh_file_path_2):
        print('[ERROR][chamfer::toChamferDistanceFromMeshFile]')
        print('\t mesh file not found!')
        print('\t mesh_file_path_2 :', mesh_file_path_2)
        return -1.0

    tag = '_sample-' + str(sample_point_num)

    pcd_file_path_1 = mesh_file_path_1.replace('.ply', tag + '.ply')
    pcd_file_path_2 = mesh_file_path_2.replace('.ply', tag + '.ply')

    if overwrite:
       removeFile(pcd_file_path_1)
       removeFile(pcd_file_path_2)

    if not os.path.exists(pcd_file_path_1):
        mesh_1 = o3d.io.read_triangle_mesh(mesh_file_path_1)
        pcd_1 = mesh_1.sample_points_poisson_disk(sample_point_num)
        o3d.io.write_point_cloud(pcd_file_path_1, pcd_1, write_ascii=True)

    if not os.path.exists(pcd_file_path_2):
        mesh_2 = o3d.io.read_triangle_mesh(mesh_file_path_2)
        pcd_2 = mesh_2.sample_points_poisson_disk(sample_point_num)
        o3d.io.write_point_cloud(pcd_file_path_2, pcd_2, write_ascii=True)

    pcd_1 = o3d.io.read_point_cloud(pcd_file_path_1)
    pcd_2 = o3d.io.read_point_cloud(pcd_file_path_2)

    points_1 = np.asarray(pcd_1.points)
    points_2 = np.asarray(pcd_2.points)

    return toChamferDistance(points_1, points_2)
