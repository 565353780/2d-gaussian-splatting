#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        get_flag = raster_settings.get_flag if hasattr(raster_settings, 'get_flag') else False
        metric_map = raster_settings.metric_map if hasattr(raster_settings, 'metric_map') and raster_settings.metric_map is not None else None
        
        # Determine which C++ function to call based on get_flag and metric_map
        # Only use metric_map version if get_flag is True AND metric_map is a valid tensor
        use_metric = False
        if get_flag and metric_map is not None:
            try:
                # Check if metric_map is a valid tensor with elements
                if isinstance(metric_map, torch.Tensor) and metric_map.numel() > 0:
                    use_metric = True
            except:
                use_metric = False
        
        # Build arguments based on whether we use metric_map
        if use_metric:
            args = (
                raster_settings.bg, 
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                metric_map,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                raster_settings.debug,
                get_flag
            )
        else:
            args = (
                raster_settings.bg, 
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                raster_settings.debug
            )
        
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                result = _C.rasterize_gaussians(*args)
                
                # 10 = with metric (accum_metric_counts, winner_id, hit_counts), 9 = no metric (winner_id, hit_counts)
                if len(result) == 10:
                    num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, accum_metric_counts, winner_id, hit_counts = result
                elif len(result) == 9:
                    num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, winner_id, hit_counts = result
                    accum_metric_counts = None
                else:
                    num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = result
                    accum_metric_counts = None
                    winner_id = None
                    hit_counts = None
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            result = _C.rasterize_gaussians(*args)
            
            if len(result) == 10:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, accum_metric_counts, winner_id, hit_counts = result
            elif len(result) == 9:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, winner_id, hit_counts = result
                accum_metric_counts = None
            else:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = result
                accum_metric_counts = None
                winner_id = None
                hit_counts = None

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        if accum_metric_counts is not None:
            return color, radii, depth, accum_metric_counts, winner_id, hit_counts
        else:
            return color, radii, depth, None, winner_id, hit_counts

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_accum_metric_counts, grad_winner_id, grad_hit_counts):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    get_flag : bool = False
    metric_map : torch.Tensor = None

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        

        # Invoke C++/CUDA rasterization routine
        result = rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
        
        # Handle return value: (color, radii, depth, accum_metric_counts, winner_id, hit_counts) - always 6 elements
        if len(result) == 6:
            return result
        else:
            return result

