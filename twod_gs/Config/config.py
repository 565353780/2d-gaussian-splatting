import os

from base_gs_trainer.Config.config import (
    GroupParams,
    ParamGroup,
    BaseModelParams,
    BasePipelineParams,
    BaseOptimizationParams,
)


def _params_to_group(params_obj) -> GroupParams:
    # 将参数对象上的属性拷贝到 GroupParams，保持与 ParamGroup.extract(args) 一致的去下划线规则
    group = GroupParams()
    for key, value in vars(params_obj).items():
        if key.startswith("_"):
            key = key[1:]
        setattr(group, key, value)
    return group


class ModelParams(BaseModelParams, ParamGroup):
    def __init__(self, parser=None, sentinel=False):
        BaseModelParams.__init__(self)

        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']

        if parser is not None:
            ParamGroup.__init__(self, parser, "Loading Parameters", sentinel)
        return

    @classmethod
    def default(cls) -> GroupParams:
        instance = cls()
        group = _params_to_group(instance)
        group.source_path = os.path.abspath(group.source_path)
        return group

class PipelineParams(BasePipelineParams, ParamGroup):
    def __init__(self, parser=None):
        BasePipelineParams.__init__(self)

        self.depth_ratio = 0.0

        if parser is not None:
            ParamGroup.__init__(self, parser, "Pipeline Parameters")
        return

    @classmethod
    def default(cls) -> GroupParams:
        instance = cls()
        return _params_to_group(instance)

class OptimizationParams(BaseOptimizationParams, ParamGroup):
    def __init__(self, parser=None):
        BaseOptimizationParams.__init__(self)
        self.opacity_lr = 0.05
        self.percent_dense = 0.01

        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.densification_interval = 100
        #self.opacity_reset_interval = 500
        self.opacity_reset_interval = 3000
        self.scaling_reset_interval = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001

        # Phase B: Surface hardening
        self.surface_ema_momentum = 0.99
        self.surface_hardening_start_iter = 1000
        self.lambda_surface = 0.01
        self.lambda_exclusive = 0.005
        self.lambda_winner_opacity = 0.05

        if parser is not None:
            ParamGroup.__init__(self, parser, "Optimization Parameters")
        return

    @classmethod
    def default(cls) -> GroupParams:
        instance = cls()
        return _params_to_group(instance)
