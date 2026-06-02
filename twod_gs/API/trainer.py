import os

from shutil import copyfile

from base_gs_trainer.Method.gpu import clear_gpu_memory, force_autograd_on
from twod_gs.Module.trainer import Trainer


def fit_gs(
    colmap_data_folder_path: str,
    gs_folder_path: str,
    gs_pcd_file_path: str,
    test_freq: int=30000,
    save_freq: int=30000,
    steps: int=30000,
) -> bool:
    """Fit a 2DGS reconstruction and emit a single ``gs.ply``.

    统一的 GS 训练 API：从 COLMAP 数据目录训练 ``steps`` 步，把最终
    ``point_cloud/iteration_{steps}/point_cloud.ply`` 复制到
    ``gs_pcd_file_path``。已有产物时直接短路返回，训练后无论成功与否都
    释放 GPU 资源。该接口与 ``fast_gs.API.trainer.fit_gs`` 完全同签名、
    同语义。
    """
    if os.path.exists(gs_pcd_file_path):
        clear_gpu_memory()
        return True

    fit_gs_pcd_file_path = os.path.join(
        gs_folder_path,
        'point_cloud/iteration_{}/point_cloud.ply'.format(steps),
    )
    if os.path.exists(fit_gs_pcd_file_path):
        copyfile(fit_gs_pcd_file_path, gs_pcd_file_path)
        clear_gpu_memory()
        return True

    trainer = None
    try:
        with force_autograd_on('fit_gs'):
            trainer = Trainer(
                colmap_data_folder_path,
                device='cuda:0',
                save_result_folder_path=gs_folder_path,
                save_log_folder_path=os.path.join(gs_folder_path, 'logs/'),
                test_freq=test_freq,
                save_freq=save_freq,
            )
            trainer.train(steps)

        copyfile(fit_gs_pcd_file_path, gs_pcd_file_path)
    finally:
        try:
            del trainer
        except Exception:
            pass
        clear_gpu_memory()
    return True
