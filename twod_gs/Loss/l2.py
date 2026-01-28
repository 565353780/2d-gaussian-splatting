import torch


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()
