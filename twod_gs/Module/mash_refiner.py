import torch

from ma_sh.Module.trainer import Trainer

class MashRefiner(object):
    def __init__(self,
                 anchor_num: int = 400,
                 mask_degree_max: int = 3,
                 sh_degree_max: int = 2,
                 mask_boundary_sample_num: int = 90,
                 sample_polar_num: int = 1000,
                 sample_point_scale: float = 0.8,
                 ) -> None:
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.sample_point_scale = sample_point_scale
        self.use_inv = True
        self.idx_dtype = torch.int64
        self.dtype = torch.float32
        self.device = "cuda:0"

        self.lr = 2e-3
        self.min_lr = 1e-3
        self.warmup_step_num = 80
        self.warmup_epoch = 4
        self.factor = 0.8
        self.patience = 2

        self.render = False
        self.render_freq = 1
        self.render_init_only = False

        self.save_result_folder_path = None
        self.save_log_folder_path = None

        self.print_progress = True
        return

    def initTrainer(self) -> bool:
        self.trainer = Trainer(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.mask_boundary_sample_num,
            self.sample_polar_num,
            self.sample_point_scale,
            self.use_inv,
            self.idx_dtype,
            self.dtype,
            self.device,
            self.lr,
            self.min_lr,
            self.warmup_step_num,
            self.warmup_epoch,
            self.factor,
            self.patience,
            self.render,
            self.render_freq,
            self.render_init_only,
            self.save_result_folder_path,
            self.save_log_folder_path,
        )
        return True

    def toSurfacePoints(self, gs_points: torch.Tensor) -> torch.Tensor:
        gs_points_array = gs_points.detach().clone().cpu().numpy()

        self.initTrainer()

        self.trainer.loadGTPoints(gs_points_array)

        self.trainer.autoTrainMash()

        with torch.no_grad():
            sample_points = self.trainer.toSamplePointsWithTransform()

        return sample_points

    def saveSurfacePointsAsPcdFile(self, save_pcd_file_path: str, overwrite: bool = False) -> bool:
        if not self.trainer.saveAsPcdFile(save_pcd_file_path, overwrite):
            print('[ERROR][MashRefiner::saveSurfacePointsAsPcdFile]')
            print('\t saveAsPcdFile failed!')
            return False

        return True
