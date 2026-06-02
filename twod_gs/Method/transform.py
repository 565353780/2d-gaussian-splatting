import torch
import numpy as np

from math import isqrt
from typing import Union

from camera_control.Method.data import toTensor
from camera_control.Method.rotate import (
    rotmat2qvec,
    decompose_similarity_from_T,
)

from twod_gs.Model.gs import GaussianModel


_SH_AXIS_PERM = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def _projectToSO3(R: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(R)
    R_proj = U @ Vt
    if torch.det(R_proj) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R_proj = U @ Vt
    return R_proj


def _rotateSHFeaturesRest(
    features_rest: torch.Tensor,
    R_left: torch.Tensor,
) -> torch.Tensor:
    '''
    对 _features_rest 按给定的左乘旋转矩阵 R_left 做球谐系数旋转。

    约定:
      - features_rest 形状: (N, (L+1)^2 - 1, 3), 最后一维为 RGB 通道
      - 按 l=1,2,3,... 分段, 每段大小 2l+1
      - 3DGS/2DGS 的 SH 约定为 yzx 次序, 需将 R_left 做 P^{-1} R P 置换到 e3nn 的 xyz 次序
      - Wigner D 使用 (alpha, -beta, gamma), 与 3DGS 社区通行实现一致:
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176
    '''
    try:
        from e3nn.o3 import matrix_to_angles, wigner_D
    except ImportError as e:
        raise ImportError(
            '[ERROR][transform::_rotateSHFeaturesRest] e3nn is required for SH rotation, '
            'please `pip install e3nn`'
        ) from e

    if features_rest.numel() == 0 or features_rest.shape[1] == 0:
        return features_rest

    device = features_rest.device
    dtype = features_rest.dtype

    with torch.no_grad():
        R_left_cpu = R_left.detach().to(device='cpu', dtype=torch.float32)
        P_cpu = _SH_AXIS_PERM.to(device='cpu', dtype=torch.float32)
        R_perm = torch.linalg.inv(P_cpu) @ R_left_cpu @ P_cpu
        R_perm = _projectToSO3(R_perm)

        alpha, beta, gamma = matrix_to_angles(R_perm)

        total_coefs = features_rest.shape[1] + 1
        max_l = isqrt(total_coefs) - 1

        rotated = features_rest.clone()
        cursor = 0
        for l in range(1, max_l + 1):
            size = 2 * l + 1
            D_l = wigner_D(l, alpha, -beta, gamma).to(device=device, dtype=dtype)

            block = rotated[:, cursor:cursor + size, :]
            block = torch.einsum('ij,njc->nic', D_l, block)
            rotated[:, cursor:cursor + size, :] = block
            cursor += size

    return rotated


def _quatMulBatched(q_world: torch.Tensor, q_gauss: torch.Tensor) -> torch.Tensor:
    '''
    用世界旋转四元数左乘每个高斯自身的旋转四元数
    约定: [w, x, y, z], 列向量左乘旋转
    q_world: (4,)
    q_gauss: (N, 4)
    返回: (N, 4)
    '''
    w1, x1, y1, z1 = q_world[0], q_world[1], q_world[2], q_world[3]
    w2 = q_gauss[:, 0]
    x2 = q_gauss[:, 1]
    y2 = q_gauss[:, 2]
    z2 = q_gauss[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)


def _applySimilarityToGaussians(
    gaussians: GaussianModel,
    R_left: torch.Tensor,
    scale: torch.Tensor,
    translate_row: torch.Tensor,
) -> None:
    '''
    将相似变换 (R_left, scale, translate_row) 原地作用于 2DGS 的 GaussianModel。

    约定:
      - R_left: (3, 3) 纯旋转矩阵 (列向量左乘, det=+1)
      - scale: 标量 (>0)
      - translate_row: (3,) 世界系下的平移, 与行向量右乘约定一致

    世界点更新:
      xyz_new = xyz @ R_left.T * scale + translate_row

    2DGS 形状更新 (与 3DGS 的关键差异在 _scaling 的维度):
      - _scaling 在 2DGS 中只有 2 个分量 (s_u, s_v), 对应椭圆盘两个切向轴的尺度,
        法向轴尺度恒为 1 不存储。各向同性相似缩放对两个切向尺度同样适用,
        故物理尺度乘 scale, 等价于对全部 2 个分量 _scaling += log(scale)。
        (此处对整张 _scaling 张量做广播加法, 对 2 维与 3 维布局均成立。)
      - _rotation 存的是四元数, build_rotation 给出 R = [t_u | t_v | normal],
        新矩阵 R_q_new = R_left @ R_q_old, 通过左乘世界旋转四元数
        q_world = rotmat2qvec(R_left) 实现, 三个轴(含法向)随场景同步旋转。
      - _features_dc (SH degree 0) 旋转不变, 不做处理
      - _features_rest (SH degree >= 1) 需用 Wigner D 按 R_left 做同步旋转,
        否则渲染时的视线方向被旋转后颜色会发生偏移
      - opacity 不变
    '''
    device = gaussians._xyz.device
    dtype = gaussians._xyz.dtype

    R_left = R_left.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype).reshape(())
    translate_row = translate_row.to(device=device, dtype=dtype).reshape(3)

    rotation_right = R_left.transpose(0, 1).contiguous()

    with torch.no_grad():
        xyz_old = gaussians._xyz.data
        xyz_new = xyz_old @ rotation_right * scale + translate_row
        gaussians._xyz.data = xyz_new

        log_scale = torch.log(scale.clamp(min=1e-12))
        gaussians._scaling.data = gaussians._scaling.data + log_scale

        q_world = rotmat2qvec(R_left).to(device=device, dtype=dtype)
        q_old = gaussians._rotation.data
        q_new = _quatMulBatched(q_world, q_old)
        gaussians._rotation.data = q_new

        if gaussians._features_rest.numel() > 0:
            gaussians._features_rest.data = _rotateSHFeaturesRest(
                gaussians._features_rest.data, R_left,
            )

    return


def _extractPureRotation(mat_3x3: torch.Tensor) -> torch.Tensor:
    '''
    用 SVD 将任意 3x3 矩阵投影到最接近的纯旋转 (det=+1) 上, 丢弃所有缩放/错切。
    '''
    U, _, Vt = torch.linalg.svd(mat_3x3)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def translateGS(
    gaussians: GaussianModel,
    translate: Union[torch.Tensor, np.ndarray, list],
) -> bool:
    '''
    将 translate 以行向量右乘约定原地作用于 gaussians:
        xyz_new = xyz + translate_row
    '''
    if gaussians is None:
        print('[ERROR][transform::translateGS]')
        print('\t gaussians is None!')
        return False

    translate_row = toTensor(translate, torch.float32, 'cpu').reshape(3)

    device = gaussians._xyz.device
    dtype = gaussians._xyz.dtype

    R_left = torch.eye(3, dtype=dtype, device=device)
    scale = torch.tensor(1.0, dtype=dtype, device=device)

    _applySimilarityToGaussians(gaussians, R_left, scale, translate_row)
    return True


def scaleGS(
    gaussians: GaussianModel,
    scale: float,
) -> bool:
    '''
    将标量 scale 原地作用于 gaussians:
        xyz_new = xyz * scale
    '''
    if gaussians is None:
        print('[ERROR][transform::scaleGS]')
        print('\t gaussians is None!')
        return False

    scale_value = float(scale)
    if scale_value <= 0:
        print('[ERROR][transform::scaleGS]')
        print('\t scale must be positive!')
        print('\t scale:', scale_value)
        return False

    device = gaussians._xyz.device
    dtype = gaussians._xyz.dtype

    R_left = torch.eye(3, dtype=dtype, device=device)
    scale_t = torch.tensor(scale_value, dtype=dtype, device=device)
    translate_row = torch.zeros(3, dtype=dtype, device=device)

    _applySimilarityToGaussians(gaussians, R_left, scale_t, translate_row)
    return True


def rotateGS(
    gaussians: GaussianModel,
    rotation_right: Union[torch.Tensor, np.ndarray, list],
) -> bool:
    '''
    rotation_right 按行向量右乘约定: xyz_new = xyz @ rotation_right
    若输入不是正交矩阵, 用 SVD 提纯到最接近的旋转 (det=+1)。
    原地作用于 gaussians。
    '''
    if gaussians is None:
        print('[ERROR][transform::rotateGS]')
        print('\t gaussians is None!')
        return False

    rotation_right_tensor = toTensor(rotation_right, torch.float32, 'cpu').reshape(3, 3)
    R_right_pure = _extractPureRotation(rotation_right_tensor)
    R_left = R_right_pure.transpose(0, 1).contiguous()

    device = gaussians._xyz.device
    dtype = gaussians._xyz.dtype

    scale = torch.tensor(1.0, dtype=dtype, device=device)
    translate_row = torch.zeros(3, dtype=dtype, device=device)

    _applySimilarityToGaussians(gaussians, R_left, scale, translate_row)
    return True


def transformGS(
    gaussians: GaussianModel,
    transform: Union[torch.Tensor, np.ndarray, list],
) -> bool:
    '''
    输入 4x4 相似变换矩阵 T_right, 采用行向量右乘约定:
        [x_new, y_new, z_new, 1] = [x, y, z, 1] @ T_right
    T_right 结构:
        T_right[:3, :3] = scale * rotation_right
        T_right[3, :3]  = translate_row
        T_right[:, 3]   = [0, 0, 0, 1]

    与 camera-control 中 CameraConvertor.transformCameras 使用同一套 similarity 分解,
    使得场景与相机在同一矩阵驱动下同步变换后, 渲染图像保持不变。
    原地作用于 gaussians。
    '''
    if gaussians is None:
        print('[ERROR][transform::transformGS]')
        print('\t gaussians is None!')
        return False

    T_right = toTensor(transform, torch.float32, 'cpu').reshape(4, 4)
    T_left = T_right.transpose(0, 1).contiguous()

    R_left, s, t_left = decompose_similarity_from_T(T_left, enforce_positive_scale=True)
    scale_safe = s.clamp(min=1e-8)

    _applySimilarityToGaussians(gaussians, R_left, scale_safe, t_left)
    return True
