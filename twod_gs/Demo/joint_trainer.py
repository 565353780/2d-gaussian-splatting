from mash_2dgs.Config.custom_path import DATA_DICTS, TEST_DATA_NAME
from mash_2dgs.Module.joint_trainer import JointTrainer

def demo(anchor_cluster_point_num: int = 400):
    data_dict = DATA_DICTS[TEST_DATA_NAME]

    source_path = data_dict['source_path']
    images = data_dict['images']
    save_result_folder_path = './output/' + TEST_DATA_NAME + '_mash-cluster-' + str(anchor_cluster_point_num) + '/'
    ply_file_path = data_dict['ply_file_path']
    conda_env_name = 'gs'
    # anchor_cluster_point_num = 400

    joint_trainer = JointTrainer(source_path, images, save_result_folder_path, ply_file_path, anchor_cluster_point_num)
    joint_trainer.train(35000)
    joint_trainer.convertToMesh(conda_env_name, 35000)
    return True
