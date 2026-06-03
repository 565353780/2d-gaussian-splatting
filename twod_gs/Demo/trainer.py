import sys
sys.path.append('../base-trainer')
sys.path.append('../base-gs-trainer')
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

from base_gs_trainer.Method.time import getCurrentTime

from twod_gs.Module.trainer import Trainer


def demo():
    data_folder_path = '/home/lichanghao/chLi/MMVideoReconV1/JJ/20260427_164113_431091/'

    colmap_data_folder_path = data_folder_path + '05_colmap_ba/'
    save_result_folder_path = data_folder_path + '06_2dgs_test/'

    trainer = Trainer(
        colmap_data_folder_path=colmap_data_folder_path,
        device='cuda:0',
        save_log_folder_path=save_result_folder_path + 'logs/' + getCurrentTime() + '/',
        save_result_folder_path=save_result_folder_path + 'results/' + getCurrentTime() + '/',
        test_freq=500,
        save_freq=500,
    )
    trainer.train(6000)
    # trainer.exportMesh()
    return True
