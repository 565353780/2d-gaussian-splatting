import sys
sys.path.append('../base-trainer')
sys.path.append('../base-gs-trainer')
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES']='6'

import cv2
import numpy as np

from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from twod_gs.Module.gs_renderer import GSRenderer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    colmap_data_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/colmap_normalized/'
    gs_ply_file_path = home + '/chLi/Dataset/GS/' + data_id + '/point_cloud.ply'
    save_normal_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/2dgs_normal/'
    os.makedirs(save_normal_folder_path, exist_ok=True)

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    fps_camera_idxs = CameraFilter.selectFPSCameras(camera_list, camera_num=10)
    fps_camera_list = [camera_list[idx] for idx in fps_camera_idxs]

    render_list = GSRenderer.renderCameras(
        gs_ply_file_path,
        fps_camera_list,
        sh_degree=3,
        bg_color=[1, 1, 1],
        device='cuda:0',
    )

    for i in range(len(fps_camera_list)):
        image_name = fps_camera_list[i].image_id

        save_normal_file_path = save_normal_folder_path + image_name

        normal = ((render_list[i]['rend_normal'].permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)

        cv2.imwrite(save_normal_file_path, normal)

    return True
