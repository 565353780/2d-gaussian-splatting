import sys
sys.path.append('../ma-sh')

import torch
import numpy as np
import open3d as o3d

from mash_2dgs.Module.mash_refiner import MashRefiner

def demo():
    mash_refiner = MashRefiner()
    gt_points_file_path = './output/00527ff4-9/point_cloud/iteration_30000/point_cloud.ply'
    save_pcd_file_path = './output/mash_fitting.ply'
    overwrite = True

    gs_pcd = o3d.io.read_point_cloud(gt_points_file_path)
    gs_points_array = np.asarray(gs_pcd.points)
    gs_points = torch.from_numpy(gs_points_array)

    surface_points = mash_refiner.toSurfacePoints(gs_points)
    print('surface_points :', surface_points.shape)

    mash_refiner.saveSurfacePointsAsPcdFile(save_pcd_file_path, overwrite)
    return True
