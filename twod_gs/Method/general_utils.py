import torch
import numpy as np


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def quaternion_from_normal(normals):
    """
    将法线向量转换为四元数 (w, x, y, z)，表示从 (0,0,1) 旋转到法线朝向的旋转。
    normals: (N, 3) 已归一化的法线，或会在内部归一化。
    returns: (N, 4) 四元数，与 build_rotation 使用的格式一致。
    """
    n = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
    nz = n[:, 2]
    nx = n[:, 0]
    ny = n[:, 1]
    # 从 (0,0,1) 到 n 的旋转：轴 axis = (0,0,1) × n = (-ny, nx, 0)，角 θ = arccos(nz)
    # q_w = cos(θ/2) = sqrt((1+nz)/2), q_xyz = sin(θ/2) * axis/|axis|
    one_plus_nz = 1.0 + nz
    eps = 1e-6
    # 当 n_z ≈ 1：恒等旋转，q = (1,0,0,0)
    # 当 n_z ≈ -1：轴退化，取 180° 绕 x 轴，q = (0,1,0,0)
    axis_len = torch.sqrt(1.0 - nz * nz).clamp(min=eps)
    sin_half = torch.sqrt((1.0 - nz).clamp(min=0) / 2.0)
    qw = torch.sqrt((one_plus_nz).clamp(min=0) / 2.0)
    qx = torch.where(one_plus_nz > eps, -ny * sin_half / axis_len, torch.ones_like(nx))
    qy = torch.where(one_plus_nz > eps, nx * sin_half / axis_len, torch.zeros_like(ny))
    qz = torch.where(one_plus_nz > eps, torch.zeros_like(nz), torch.zeros_like(nz))
    qw = torch.where(one_plus_nz < eps, torch.zeros_like(qw), qw)
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # Normalize the batch of direction vectors
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    # Create a batch of arbitrary vectors that are not collinear with the direction vectors
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(direction_vectors.device).expand(direction_vectors.shape[0], -1).clone()
    is_collinear = torch.all(torch.abs(direction_vectors - v1) < 1e-5, dim=-1)
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(direction_vectors.device)

    # Calculate the first orthogonal vectors
    v1 = torch.cross(direction_vectors, v1)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True))
    # Calculate the second orthogonal vectors by taking the cross product
    v2 = torch.cross(direction_vectors, v1)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True))
    # Create the batch of rotation matrices with the direction vectors as the last columns
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices
