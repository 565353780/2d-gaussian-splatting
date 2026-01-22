from arguments import ParamGroup

class JointOptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.position_lr_max_steps = 35000
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.percent_dense = 0.01

        self.lambda_dssim = 0.2
        self.lambda_dist = 100000.0
        self.lambda_normal = 0.01
        self.lambda_opacity = 0.001
        self.lambda_scaling = 0.001
        self.lambda_surface = 0.001
        self.opacity_cull = 0.05

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.mash_refine_interval = 5000
        self.densify_from_iter = 500
        self.densify_until_iter = -1
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Joint Optimization Parameters")
