from base_gs_trainer.Method.filter import removeFloatGSGeneric

from twod_gs.Model.gs import GaussianModel


def removeFloatGS(gs: GaussianModel, **kwargs) -> bool:
    return removeFloatGSGeneric(gs, **kwargs)


__all__ = ['removeFloatGS']
