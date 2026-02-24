/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Register both overloads with explicit casting
  m.def("rasterize_gaussians", 
    [](const torch::Tensor& background,
       const torch::Tensor& means3D,
       const torch::Tensor& colors,
       const torch::Tensor& opacity,
       const torch::Tensor& scales,
       const torch::Tensor& rotations,
       const float scale_modifier,
       const torch::Tensor& transMat_precomp,
       const torch::Tensor& viewmatrix,
       const torch::Tensor& projmatrix,
       const float tan_fovx, 
       const float tan_fovy,
       const int image_height,
       const int image_width,
       const torch::Tensor& sh,
       const int degree,
       const torch::Tensor& campos,
       const bool prefiltered,
       const bool debug) {
      return RasterizeGaussiansCUDA(
          background, means3D, colors, opacity, scales, rotations,
          scale_modifier, transMat_precomp, viewmatrix, projmatrix,
          tan_fovx, tan_fovy, image_height, image_width,
          sh, degree, campos, prefiltered, debug);
    },
    "Rasterize gaussians without metric_map");
  
  m.def("rasterize_gaussians",
    [](const torch::Tensor& background,
       const torch::Tensor& means3D,
       const torch::Tensor& colors,
       const torch::Tensor& opacity,
       const torch::Tensor& scales,
       const torch::Tensor& rotations,
       const float scale_modifier,
       const torch::Tensor& transMat_precomp,
       const torch::Tensor& metric_map,
       const torch::Tensor& viewmatrix,
       const torch::Tensor& projmatrix,
       const float tan_fovx, 
       const float tan_fovy,
       const int image_height,
       const int image_width,
       const torch::Tensor& sh,
       const int degree,
       const torch::Tensor& campos,
       const bool prefiltered,
       const bool debug,
       const bool get_flag) {
      return RasterizeGaussiansCUDA(
          background, means3D, colors, opacity, scales, rotations,
          scale_modifier, transMat_precomp, metric_map, viewmatrix, projmatrix,
          tan_fovx, tan_fovy, image_height, image_width,
          sh, degree, campos, prefiltered, debug, get_flag);
    },
    "Rasterize gaussians with metric_map");
  
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}