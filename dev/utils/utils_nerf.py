import torch
import numpy as np
import scipy
from numpy import ndarray


def generate_rays_numpy(poses: np.ndarray, focals: np.ndarray, width: int, height: int, randomize: bool):
    """
    generate rays for all cameras, slow version (numpy)

    Args:
    - poses: np.ndarray of shape (#cameras, 4, 4)
    - focals: np.ndarray of shape (#cameras)
    - width: int
    - height: int
    - randomize: bool

    Returns:
    - rays_o: np.ndarray of shape (#cameras, H, W, 3)
    - rays_d: np.ndarray of shape (#cameras, H, W, 3)
    - du: np.ndarray of shape (H, W)
    - dv: np.ndarray of shape (H, W)
    """
    dtype = poses.dtype

    u, v = np.meshgrid(np.linspace(0, width - 1, width, dtype=dtype), np.linspace(0, height - 1, height, dtype=dtype), indexing='xy')  # (H, W), (H, W)

    if randomize:
        du, dv = np.random.rand(*u.shape), np.random.rand(*v.shape)  # (H, W), (H, W)
        u, v = np.clip(u + du, 0, width - 1), np.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
    u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)

    dirs = np.stack([u_normalized, -v_normalized, -np.ones_like(u_normalized)], axis=-1)  # (N, H, W, 3)
    rays_d = np.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = np.broadcast_to(poses[:, None, None, :3, 3], (poses.shape[0], height, width, 3))  # (N, H, W, 3)

    return rays_o, rays_d, u, v


def generate_rays_device(poses: torch.Tensor, focals: torch.Tensor, width: int, height: int, randomize: bool):
    """
    generate rays for all cameras, fast version (torch)

    Args:
    - poses: torch.Tensor of shape (#cameras, 4, 4)
    - focals: torch.Tensor of shape (#cameras)
    - width: int
    - height: int
    - randomize: bool

    Returns:
    - rays_o: torch.Tensor of shape (#cameras, H, W, 3)
    - rays_d: torch.Tensor of shape (#cameras, H, W, 3)
    - du: torch.Tensor of shape (H, W)
    - dv: torch.Tensor of shape (H, W)
    """
    device = poses.device
    dtype = poses.dtype

    u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=device, dtype=dtype), torch.linspace(0, height - 1, height, device=device, dtype=dtype), indexing='xy')  # (H, W), (H, W)

    if randomize:
        du, dv = torch.rand_like(u), torch.rand_like(v)  # (H, W), (H, W)
        u, v = torch.clip(u + du, 0, width - 1), torch.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
    u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)

    dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)
    rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = poses[:, None, None, :3, 3].expand(-1, height, width, -1)  # (N, H, W, 3)

    return rays_o, rays_d, u, v


def get_points_device(rays_o: torch.Tensor, rays_d: torch.Tensor, near: float, far: float, N_depths: int, randomize: bool):
    # 在 [near, far] 之间等间隔采样
    depths = torch.linspace(near, far, steps=N_depths, device=rays_o.device).expand(rays_o.shape[0], N_depths)  # (N, N_depths)
    depths_target = depths.clone()

    # 如果启用随机抖动，增加噪声
    if randomize:
        midpoints = (depths[:, :-1] + depths[:, 1:]) / 2.0  # 计算相邻点的中点
        noise = (torch.rand_like(midpoints) - 0.5) * (far - near) / N_depths  # 在每个区间内随机扰动
        depths_target[:, :-1] = midpoints + noise  # 只修改前 N-1 个点

    # 计算采样点的位置
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * depths_target.unsqueeze(-1)  # (N, N_depths, 3)

    return points, depths_target  # 返回采样点和深度值


def resample_images_scipy(images: ndarray, u: ndarray, v: ndarray):
    """
    Resample images using bilinear interpolation

    Args:
    - images: np.ndarray of shape (..., H, W, C)
    - u: np.ndarray of shape (H, W)
    - v: np.ndarray of shape (H, W)

    Returns:
    - resampled_images: np.ndarray of shape (..., H, W, C)
    """
    H, W, C = images.shape[-3:]

    orig_shape = images.shape  # original shape
    reshaped_images = images.reshape(-1, H, W, C)  # (B, H, W, C)
    resampled_images = np.zeros_like(reshaped_images)  # (B, H, W, C)

    for batch in range(reshaped_images.shape[0]):
        for c in range(reshaped_images.shape[-1]):
            img = reshaped_images[batch, :, :, c]  # (H, W)
            ret: np.ndarray = scipy.ndimage.map_coordinates(img, [v.ravel(), u.ravel()], order=1, mode='nearest')  # (H * W)
            resampled_images[batch, :, :, c] = ret.reshape(H, W)  # (H, W)

    return resampled_images.reshape(orig_shape)  # restore original shape


def resample_images_torch(images: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """
    使用 PyTorch 进行双线性插值，基于 u, v 重新采样图像

    参数:
    - images: torch.Tensor, 形状 (..., H, W, C)，原始图像数据
    - u: torch.Tensor, 形状 (H, W)，目标 x 位置（列）
    - v: torch.Tensor, 形状 (H, W)，目标 y 位置（行）

    返回:
    - resampled_images: torch.Tensor, 形状和 `images` 相同，经过插值后的图像
    """
    H, W, C = images.shape[-3:]

    # **归一化 (u, v) 到 [-1, 1]**
    u_norm, v_norm = 2.0 * (u / (W - 1)) - 1, 2.0 * (v / (H - 1)) - 1

    # 组合成 `grid_sample` 需要的 grid，形状 (1, H, W, 2)
    grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # 调整 images 形状：从 (..., H, W, C) → (batch, C, H, W)
    orig_shape = images.shape
    reshaped_images = images.reshape(-1, H, W, C).permute(0, 3, 1, 2)  # (batch, C, H, W)

    # 使用 grid_sample 进行插值
    resampled = torch.nn.functional.grid_sample(reshaped_images, grid.expand(reshaped_images.shape[0], -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)

    # 恢复回原始形状：从 (batch, C, H, W) → (..., H, W, C)
    resampled_images = resampled.permute(0, 2, 3, 1).reshape(orig_shape)

    return resampled_images
