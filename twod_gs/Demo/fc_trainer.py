import sys
sys.path.append('../base-trainer')
sys.path.append('../base-gs-trainer')
sys.path.append('../camera-control')
sys.path.append('../flexi-cubes')
sys.path.append('../mv-fc-recon')

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

from base_gs_trainer.Method.time import getCurrentTime

from twod_gs.Module.fc_trainer import FCTrainer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    colmap_data_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/colmap_normalized/'
    init_mesh_file_path = home + '/chLi/Dataset/GS/' + data_id + '/4_n_stage1_mv.ply'
    save_result_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/2dgs/'

    fc_trainer = FCTrainer(
        colmap_data_folder_path=colmap_data_folder_path,
        init_mesh_file_path=init_mesh_file_path,
        device='cuda:0',
        save_log_folder_path=save_result_folder_path + 'logs/' + getCurrentTime() + '/',
        save_result_folder_path=save_result_folder_path + 'results/' + getCurrentTime() + '/',
        test_freq=500,
        save_freq=500,
        fc_update_freq=500,
    )
    fc_trainer.train(30000)
    return True
