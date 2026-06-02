from base_gs_trainer.Method.io import loadGSByClass, saveGS

from twod_gs.Model.gs import GaussianModel


def loadGS(gs_ply_file_path: str, sh_degree: int = 3) -> GaussianModel:
    return loadGSByClass(GaussianModel, gs_ply_file_path, sh_degree=sh_degree)


__all__ = ['loadGS', 'saveGS']
