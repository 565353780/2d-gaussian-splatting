import sys
sys.path.append('../base-trainer')

from mash_2dgs.Config.custom_path import DATA_DICTS, TEST_DATA_NAME
from mash_2dgs.Module.trainer import Trainer


def demo():
    data_dict = DATA_DICTS[TEST_DATA_NAME]

    source_path = data_dict['source_path']
    images = data_dict['images']
    save_result_folder_path = './output/' + TEST_DATA_NAME + '_mashgs/'
    save_result_folder_path = 'auto'
    conda_env_name = 'gs'

    trainer = Trainer(source_path, images, save_result_folder_path)
    trainer.train(35000)
    trainer.convertToMesh(conda_env_name, 35000)
    return True
