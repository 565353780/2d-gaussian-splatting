import sys
sys.path.append('../base-trainer')

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

from twod_gs.Module.trainer import Trainer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    source_path = home + '/chLi/Dataset/GS/' + data_id + '/gs/'
    save_result_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/2dgs/'

    trainer = Trainer(
        source_path=source_path,
        save_log_folder_path=save_result_folder_path + 'log/',
        save_result_folder_path=save_result_folder_path + 'results/',
    )
    trainer.train(35000)
    trainer.convertToMesh(conda_env_name, 35000)
    return True
