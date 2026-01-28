import sys
sys.path.append('../base-trainer')

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from twod_gs.Method.time import getCurrentTime
from twod_gs.Module.trainer import Trainer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    colmap_data_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/gs/'
    save_result_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/2dgs/'

    trainer = Trainer(
        colmap_data_folder_path=colmap_data_folder_path,
        save_log_folder_path=save_result_folder_path + 'logs/' + getCurrentTime() + '/',
        save_result_folder_path=save_result_folder_path + 'results/' + getCurrentTime() + '/',
    )
    trainer.train(10000)
    trainer.exportMesh()
    return True
