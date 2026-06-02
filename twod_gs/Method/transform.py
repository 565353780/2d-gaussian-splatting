import numpy as np
import torch

from typing import Union

from base_gs_trainer.Method.transform import (
    translateGSGeneric,
    scaleGSGeneric,
    rotateGSGeneric,
    transformGSGeneric,
)

from twod_gs.Model.gs import GaussianModel


def translateGS(
    gaussians: GaussianModel,
    translate: Union[torch.Tensor, np.ndarray, list],
) -> bool:
    return translateGSGeneric(gaussians, translate)


def scaleGS(
    gaussians: GaussianModel,
    scale: float,
) -> bool:
    return scaleGSGeneric(gaussians, scale)


def rotateGS(
    gaussians: GaussianModel,
    rotation_right: Union[torch.Tensor, np.ndarray, list],
) -> bool:
    return rotateGSGeneric(gaussians, rotation_right)


def transformGS(
    gaussians: GaussianModel,
    transform: Union[torch.Tensor, np.ndarray, list],
) -> bool:
    return transformGSGeneric(gaussians, transform)


__all__ = ['translateGS', 'scaleGS', 'rotateGS', 'transformGS']
