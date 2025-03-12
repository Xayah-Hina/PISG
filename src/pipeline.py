import torch
import torchvision.io as io
import torch.multiprocessing as mp
import os
import math
from pathlib import Path

from model.model_hyfluid import NeRFSmall
from model.encoder_hyfluid import HashEncoderNative

"""
Performance:

- 1024 + 1.0 - 8100it, 62.30it/s, 134s, 37.4GB

- 2048 + 1.0 - 4050it, 37.40it/s, 110s, 37.4GB
- 512 + 1.0 - 16200it, 90.26it/s, , 181s, 37.4GB
- 256 + 1.0 - 32400it, 128.83it/s, 254s, 37.4GB
- 128 + 1.0 - 64800it, 142.91it/s, 461s, 37.4GB

- 1024 + 0.5 - 2025it, 66.37it/s, 45s, 12.5GB

- 1024 + 1.0 ( with mask filter ): 3721it, 62.70it/s, 100s, 37.4GB

"""


def find_relative_paths(relative_path_list):
    current_dir = Path.cwd()
    search_dirs = [current_dir, current_dir.parent, current_dir.parent.parent]

    for i in range(len(relative_path_list)):
        found = False
        relative_path = relative_path_list[i]
        for directory in search_dirs:
            full_path = directory / relative_path
            if full_path.exists():
                relative_path_list[i] = str(full_path.resolve())
                found = True
                break

        if not found:
            raise FileNotFoundError(f"file not found: {relative_path}")


# Our Scene 1
training_videos_scene1 = [
    "data/PISG/scene1/front.mp4",
    "data/PISG/scene1/right.mp4",
    "data/PISG/scene1/back.mp4",
    "data/PISG/scene1/top.mp4",
    "data/PISG/scene1/bottom.mp4",
]

camera_calibrations_scene1 = [
    "data/PISG/scene1/cam_front.npz",
    "data/PISG/scene1/cam_right.npz",
    "data/PISG/scene1/cam_back.npz",
    "data/PISG/scene1/cam_top.npz",
    "data/PISG/scene1/cam_bottom.npz",
]
scene_min_scene1 = [-20.0, -20.0, -20.0]
scene_max_scene1 = [20.0, 20.0, 20.0]

# HyFluid Scene
training_videos_hyfluid = [
    "data/hyfluid/train00.mp4",
    "data/hyfluid/train01.mp4",
    "data/hyfluid/train02.mp4",
    "data/hyfluid/train03.mp4",
    "data/hyfluid/train04.mp4",
]
scene_min_hyfluid = [-1, -1, -1]
scene_max_hyfluid = [1, 1, 1]

camera_calibrations_hyfluid = [
    "data/hyfluid/cam_train00.npz",
    "data/hyfluid/cam_train01.npz",
    "data/hyfluid/cam_train02.npz",
    "data/hyfluid/cam_train03.npz",
    "data/hyfluid/cam_train04.npz",
]

training_videos = training_videos_hyfluid
camera_calibrations = camera_calibrations_hyfluid
scene_min_current = scene_min_hyfluid
scene_max_current = scene_max_hyfluid

find_relative_paths(training_videos)
find_relative_paths(camera_calibrations)


def resample_frames(frames: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """
    Resample frames using the given UV coordinates.

    Args:
    - frames: torch.Tensor of shape (..., H, W, C)
    - u: torch.Tensor of shape (N, H, W)
    - v: torch.Tensor of shape (N, H, W)

    Returns:
    - resampled_images: torch.Tensor of shape (N, T, H, W, C)
    """

    H, W, C = frames.shape[-3:]
    u_norm, v_norm = 2.0 * (u / (W - 1)) - 1, 2.0 * (v / (H - 1)) - 1
    grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)  # (1, H, W, 2)
    orig_shape = frames.shape
    reshaped_images = frames.reshape(-1, H, W, C).permute(0, 3, 1, 2)  # (batch, C, H, W)
    resampled = torch.nn.functional.grid_sample(reshaped_images, grid.expand(reshaped_images.shape[0], -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)
    resampled_images = resampled.permute(0, 2, 3, 1).reshape(orig_shape)
    return resampled_images


def sample_frustum(dirs: torch.Tensor, poses: torch.Tensor, near: float, far: float, depth: int, batch_size: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    """
    Sample points in the frustum of each camera.

    Args:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - poses: torch.Tensor of shape (N, 4, 4)
    - near: float
    - far: float
    - depth: int
    - batch_size: int
    - randomize: bool

    Yields:
    - points: torch.Tensor of shape (batch_size, depth, 3)
    """

    rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # (N, H, W, 3)

    rays_d = rays_d.reshape(-1, 3)  # (N*H*W, 3)
    rays_o = rays_o.reshape(-1, 3)  # (N*H*W, 3)
    num_rays = rays_d.shape[0]

    depths = torch.linspace(near, far, steps=depth, device=device, dtype=dtype).unsqueeze(0)  # (1, depth)

    if randomize:
        indices = torch.randperm(num_rays, device=device)  # (N*H*W)
    else:
        indices = torch.arange(num_rays, device=device)  # (N*H*W)

    for i in range(0, num_rays, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_rays_o = rays_o[batch_indices]  # (batch_size, 3)
        batch_rays_d = rays_d[batch_indices]  # (batch_size, 3)

        batch_depths = depths.clone()

        if randomize:
            midpoints = (depths[:, :-1] + depths[:, 1:]) / 2.0
            noise = (torch.rand_like(midpoints) - 0.5) * (far - near) / depth
            batch_depths[:, :-1] = midpoints + noise

        batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * batch_depths[:, :, None]  # (batch_size, depth, 3)
        yield batch_points, batch_depths, batch_indices


def sample_frustum_with_mask(dirs: torch.Tensor, poses: torch.Tensor, mask: torch.Tensor, near: float, far: float, depth: int, batch_size: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    """
    Sample points in the frustum of each camera with mask filtering.

    Args:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - poses: torch.Tensor of shape (N, 4, 4)
    - mask: torch.Tensor of shape (N, 4)
    - near: float
    - far: float
    - depth: int
    - batch_size: int
    - randomize: bool

    Yields:
    - points: torch.Tensor of shape (batch_size, depth, 3)
    """

    N, H, W = dirs.shape[:3]
    num_origin = N * H * W

    rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # (N, H, W, 3)

    rays_d = run_filter1(rays_d, mask)
    rays_o = run_filter1(rays_o, mask)

    indices_origin = torch.arange(num_origin, device=device)  # (N*H*W)
    indices_origin_shaped = indices_origin.reshape(N, H, W, 1)
    indices_filtered = run_filter1(indices_origin_shaped, mask).flatten()  # (filtered)

    num_filtered = rays_d.shape[0]
    if randomize:
        indices_inner = torch.randperm(num_filtered, device=device)  # (N*H*W)
    else:
        indices_inner = torch.arange(num_filtered, device=device)  # (N*H*W)

    depths = torch.linspace(near, far, steps=depth, device=device, dtype=dtype).unsqueeze(0)  # (1, depth)
    for i in range(0, num_filtered, batch_size):
        idx = indices_inner[i:i + batch_size]
        batch_rays_o = rays_o[idx]  # (batch_size, 3)
        batch_rays_d = rays_d[idx]  # (batch_size, 3)

        batch_depths = depths.clone()
        if randomize:
            midpoints = (depths[:, :-1] + depths[:, 1:]) / 2.0
            noise = (torch.rand_like(midpoints) - 0.5) * (far - near) / depth
            batch_depths[:, :-1] = midpoints + noise

        batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * batch_depths[:, :, None]  # (batch_size, depth, 3)
        batch_indices = indices_filtered[idx]
        yield batch_points, batch_depths, batch_indices


def sample_random_frame(videos_data: torch.Tensor, batch_indices: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """
    Sample a random frame from the given videos data.

    Args:
    - videos_data: torch.Tensor of shape (T, V, H, W, C)
    - batch_indices: torch.Tensor of shape (batch_size)

    Returns:
    - batch_time: torch.Tensor of shape (1)
    - batch_target_pixels: torch.Tensor of shape (batch_size, C)
    """
    frame = torch.rand((), device=device, dtype=dtype) * (videos_data.shape[0] - 1)
    frame_floor = torch.floor(frame).long()
    frame_ceil = frame_floor + 1
    frames_alpha = frame - frame_floor.to(frame.dtype)
    target_frame = (1 - frames_alpha) * videos_data[frame_floor] + frames_alpha * videos_data[frame_ceil]  # (V * H * W, C)
    target_frame = target_frame.reshape(-1, 3)
    batch_target_pixels = target_frame[batch_indices]  # (batch_size, C)
    batch_time = frame / (videos_data.shape[0] - 1)

    return batch_time, batch_target_pixels


def normalize_points(points: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """
    Normalize the points to the range [0, 1].

    Args:
    - points: torch.Tensor of shape (..., 3)

    Returns:
    - points_normalized: torch.Tensor of shape (..., 3)
    """

    scene_min = torch.tensor(scene_min_current, device=device, dtype=dtype)
    scene_max = torch.tensor(scene_max_current, device=device, dtype=dtype)
    points_normalized = (points - scene_min) / (scene_max - scene_min)
    return points_normalized


def shuffle_uv(focals: torch.Tensor, width: int, height: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    """
    Shuffle the UV coordinates of the image plane.

    Args:
    - focals: torch.Tensor of shape (N)
    - width: int
    - height: int
    - randomize: bool

    Returns:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - u: torch.Tensor of shape (H, W)
    - v: torch.Tensor of shape (H, W)
    """
    focals = focals.to(device)
    u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=device, dtype=dtype), torch.linspace(0, height - 1, height, device=device, dtype=dtype), indexing='xy')  # (H, W), (H, W)
    if randomize:
        du, dv = torch.rand_like(u), torch.rand_like(v)  # (H, W), (H, W)
        u, v = torch.clip(u + du, 0, width - 1), torch.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
    u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)
    dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)

    return dirs, u, v


def get_filter_mask(video_tensor):
    # 1. 对通道做any运算，判断每个像素在一帧内是否非0
    mask_tensor = (video_tensor != 0)
    frame_mask = mask_tensor.any(dim=-1)  # shape: [120, 4, 1920, 1080]

    # 2. 对帧做any运算，得到每个视频中每个像素是否曾非0
    video_mask = frame_mask.any(dim=0)  # shape: [4, 1920, 1080]

    # 3. 对于每个视频计算包围矩形
    bboxes = []
    for mask in video_mask:
        coords = mask.nonzero()
        if coords.numel() == 0:
            raise AssertionError("All frames are zero.")
        # nonzero 返回的坐标顺序为 (row, col)，即 (y, x)
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values
        # 调整顺序为 [x_min, x_max, y_min, y_max]
        bbox = torch.stack([x_min, x_max, y_min, y_max])
        bboxes.append(bbox)
    # 堆叠后得到形状为 [videos, 4]
    return torch.stack(bboxes, dim=0)


def run_filter1(dirs: torch.Tensor, mask: torch.Tensor):
    """
    Args:
        dirs: torch.Tensor，形状为 [4, 1920, 1080, 3]，每个视频的方向数据
        mask: torch.Tensor，形状为 [4, 4]，每一行格式为 [x_min, x_max, y_min, y_max]

    Returns:
        flattened_dirs: torch.Tensor，形状为 (X, 3)，包含所有视频在对应 bounding box 内的方向数据，经过扁平化
    """
    flattened_list = []
    for d, bbox in zip(dirs, mask):
        # mask 的每一行为 [x_min, x_max, y_min, y_max]
        x_min, x_max, y_min, y_max = bbox.tolist()
        # 注意这里对索引转换为 int，并加 1 以确保包含上界
        filtered = d[int(y_min):int(y_max) + 1, int(x_min):int(x_max) + 1, :]
        # 扁平化为 (N, 3) 的形状
        flattened = filtered.reshape(-1, filtered.shape[-1])
        flattened_list.append(flattened)
    # 将所有视频的扁平化 tensor 在第一维上拼接起来
    flattened_dirs = torch.cat(flattened_list, dim=0)
    return flattened_dirs


def is_points_in_frustum(points, camera_poses, focals, width, height, near, far):
    """
    判断 M 个 3D 点是否在 N 个相机的视锥体内
    points: (M, 3)  ->  M 个 3D 点
    camera_poses: (N, 4, 4)  ->  N 个相机的 4x4 位姿矩阵
    focals, width, height, near, far: (N,)  ->  每个相机的参数
    返回: (M, N)  ->  M 个点是否在 N 个相机视锥体内的布尔张量
    """

    def perspective_matrix(focals, width, height, near, far):
        """ 计算 N 个相机的透视投影矩阵 (N, 4, 4) """
        N = focals.shape[0]
        P = torch.zeros((N, 4, 4), device=focals.device)

        P[:, 0, 0] = 2 * focals / width
        P[:, 1, 1] = 2 * focals / height
        P[:, 2, 2] = -(far + near) / (far - near)
        P[:, 2, 3] = -2 * far * near / (far - near)
        P[:, 3, 2] = -1

        return P

    N = camera_poses.shape[0]
    M = points.shape[0]

    # 计算 N 个透视投影矩阵 (N, 4, 4)
    P = perspective_matrix(focals, width, height, near, far)

    # 计算 N 个视图矩阵 (N, 4, 4)
    V = torch.inverse(camera_poses)

    # 扩展 M 个点为 (M, 4) 的齐次坐标
    ones = torch.ones((M, 1), device=points.device)
    points_h = torch.cat([points, ones], dim=-1).unsqueeze(0)  # (1, M, 4)

    # 变换点到相机坐标系 (N, M, 4)
    points_cam = torch.einsum('nij,mj->nmi', V, points_h.squeeze(0))

    # 变换点到裁剪空间 (N, M, 4)
    points_clip = torch.einsum('nij,nmj->nmi', P, points_cam)

    # 进行透视除法 (N, M, 3)
    points_ndc = points_clip[:, :, :3] / points_clip[:, :, 3].unsqueeze(-1)

    # 检查是否在 NDC 空间 (-1, 1) 内 (M, N)
    in_frustum = (
            (points_ndc[:, :, 0] >= -1) & (points_ndc[:, :, 0] <= 1) &  # -1 ≤ x ≤ 1
            (points_ndc[:, :, 1] >= -1) & (points_ndc[:, :, 1] <= 1) &  # -1 ≤ y ≤ 1
            (points_ndc[:, :, 2] >= -1) & (points_ndc[:, :, 2] <= 1)  # -1 ≤ z ≤ 1
    ).T  # 转置以匹配 (M, N)

    return in_frustum


def sample_points(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, res: int, device: torch.device, dtype: torch.dtype):
    xs, ys, zs = torch.meshgrid([torch.linspace(x_min, x_max, res, device=device, dtype=dtype), torch.linspace(y_min, y_max, res, device=device, dtype=dtype), torch.linspace(z_min, z_max, res, device=device, dtype=dtype)], indexing='ij')
    grids = torch.stack([xs, ys, zs], dim=-1)
    return grids


class PISGPipelineTorch:
    """
    Pipeline for training and testing the PISG model using PyTorch.
    """

    def __init__(self, torch_device: torch.device, torch_dtype: torch.dtype):
        self.device = torch_device
        self.dtype = torch_dtype
        self.encoder_num_scale = 16
        self.depth = 192
        self.ratio = 0.5

        self.encoder = HashEncoderNative(device=self.device).to(self.device)
        self.model = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_num_scale * 2).to(self.device)
        self.optimizer = torch.optim.RAdam([{'params': self.model.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

        target_lr_ratio = 0.1  # 新增：使用渐进式指数衰减，设定训练结束时的学习率为初始学习率的 10% 你可以根据需求调整目标比例
        gamma = math.exp(math.log(target_lr_ratio) / 30000)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        def volume_render(batch_points: torch.Tensor, batch_depths: torch.Tensor, batch_time: torch.Tensor, poses: torch.Tensor, focals: torch.Tensor, width: torch.Tensor, height: torch.Tensor, near: torch.Tensor, far: torch.Tensor):
            batch_points_flat = batch_points.reshape(-1, 3)  # (batch_size * depth, 3)
            in_frustum_mask = is_points_in_frustum(points=batch_points_flat, camera_poses=poses, focals=focals, width=width, height=height, near=near, far=far)
            in_all_frustum_mask = torch.all(in_frustum_mask, dim=1)
            batch_points_normalized_flat = normalize_points(points=batch_points_flat, device=self.device, dtype=self.dtype)  # (batch_size * depth, 3)
            batch_input_xyzt_flat = torch.cat([batch_points_normalized_flat, batch_time.expand(batch_points_normalized_flat[..., :1].shape)], dim=-1).reshape(-1, 4)  # (batch_size * depth, 4)
            batch_rgb_map = self.query_rgb_map_skip_ghost(batch_input_xyzt_flat=batch_input_xyzt_flat, batch_depths=batch_depths, in_all_frustum_mask=in_all_frustum_mask)
            return batch_rgb_map

        self.compiled_volume_render = torch.compile(volume_render, mode="max-autotune")

        def query_density_grid(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, res: int, time: float, poses: torch.Tensor, focals: torch.Tensor, width: torch.Tensor, height: torch.Tensor, near: torch.Tensor, far: torch.Tensor):
            with torch.no_grad():
                xyz = sample_points(x_min, x_max, y_min, y_max, z_min, z_max, res, device=self.device, dtype=self.dtype)  # (res, res, res, 3)
                xyz_flat = xyz.reshape(-1, 3)  # (res * res * res, 3)
                in_frustum_mask = is_points_in_frustum(points=xyz_flat, camera_poses=poses, focals=focals, width=width, height=height, near=near, far=far)  # (res * res * res, N)
                in_all_frustum_mask = torch.all(in_frustum_mask, dim=1)  # (res * res * res)
                xyz_normalized_flat = normalize_points(points=xyz_flat, device=self.device, dtype=self.dtype)  # (res * res * res, 3)
                input_xyzt_flat = torch.cat([xyz_normalized_flat, torch.ones_like(xyz_normalized_flat[..., :1]) * time], dim=-1).reshape(-1, 4)  # (res * res * res, 4)
                # batchfy this
                raw_d_flat_list = []
                batch_size = 64 * 64 * 64
                for i in range(0, input_xyzt_flat.shape[0], batch_size):
                    input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                    raw_d_flat_batch = self.model(self.encoder(input_xyzt_flat_batch))
                    raw_d_flat_list.append(raw_d_flat_batch)
                raw_d_flat = torch.cat(raw_d_flat_list, dim=0)  # (res * res * res, 1)
                in_all_frustum_mask_float = in_all_frustum_mask.unsqueeze(-1).float()  # (res * res * res, 1)
                raw_d_flat = raw_d_flat * in_all_frustum_mask_float
                raw_d = raw_d_flat.reshape(res, res, res, 1)  # (res, res, res, 1)
                return raw_d

        self.compiled_query_density_grid = torch.compile(query_density_grid, mode="max-autotune")

    def train(self, batch_size, save_ckp_path):
        videos_data = self.load_videos_data(*training_videos, ratio=self.ratio)  # (T, V, H, W, C)
        masks = get_filter_mask(video_tensor=videos_data)
        poses, focals, width, height, near, far = self.load_cameras_data(*camera_calibrations)
        focals = focals * self.ratio
        width = width * self.ratio
        height = height * self.ratio

        import tqdm
        for _1 in tqdm.trange(0, 5):
            losses = []  # 记录 loss
            temp_losses = []  # 用于计算平均 loss

            dirs, u, v = shuffle_uv(focals=focals, width=int(width[0].item()), height=int(height[0].item()), randomize=True, device=torch.device("cpu"), dtype=self.dtype)
            videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(self.device)  # (T, V, H, W, C)
            dirs = dirs.to(self.device)

            for _2, (batch_points, batch_depths, batch_indices) in enumerate(sample_frustum_with_mask(dirs=dirs, poses=poses, mask=masks, near=near[0].item(), far=far[0].item(), depth=self.depth, batch_size=batch_size, randomize=True, device=self.device, dtype=self.dtype)):
                batch_time, batch_target_pixels = sample_random_frame(videos_data=videos_data_resampled, batch_indices=batch_indices, device=self.device, dtype=self.dtype)  # (batch_size, C)
                batch_rgb_map = self.compiled_volume_render(batch_points, batch_depths, batch_time, poses, focals, width, height, near, far)
                loss_image = torch.nn.functional.mse_loss(batch_rgb_map, batch_target_pixels)
                self.optimizer.zero_grad()
                loss_image.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 记录 loss
                temp_losses.append(loss_image.item())

                # 每 100 次计算平均 loss
                if (_2 + 1) % 100 == 0:
                    avg_loss = sum(temp_losses) / len(temp_losses)
                    losses.append(avg_loss)
                    allocated_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                    reserved_mem = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB
                    tqdm.tqdm.write(f"Iteration {_2 + 1}: Avg Loss = {avg_loss:.6f} | Allocated Mem: {allocated_mem:.2f} MB | Reserved Mem: {reserved_mem:.2f} MB")
                    temp_losses.clear()  # 清空

            import matplotlib.pyplot as plt

            # 训练结束后绘制 loss 曲线
            plt.figure(figsize=(10, 5))
            plt.plot(range(100, 100 * len(losses) + 1, 100), losses, marker='o', linestyle='-')
            plt.xlabel("Iterations")
            plt.ylabel("Average Loss")
            plt.title(f"Training Loss Over Iterations: {_1}")
            plt.grid()
            plt.show()

        if save_ckp_path is not None:
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'model_state_dict': self.model.state_dict(),
            }, save_ckp_path)

    def test(self, save_ckp_path, target_timestamp: int, output_dir="output"):
        import numpy as np
        import imageio.v3 as imageio
        os.makedirs(output_dir, exist_ok=True)

        ckpt = torch.load(save_ckp_path)
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.model.load_state_dict(ckpt['model_state_dict'])

        poses, focals, width, height, near, far = self.load_cameras_data(*camera_calibrations)

        batch_size = 1024
        test_timestamp = torch.tensor(target_timestamp / 120., device=self.device, dtype=self.dtype)
        import tqdm
        with torch.no_grad():
            dirs, _, _ = shuffle_uv(focals=focals[:1], width=int(width[0].item()), height=int(height[0].item()), randomize=False, device=self.device, dtype=self.dtype)
            rgb_map_list = []
            for _1, (batch_points, batch_depths, batch_indices) in enumerate(tqdm.tqdm(sample_frustum(dirs=dirs, poses=poses[:1], near=near[0].item(), far=far[0].item(), depth=self.depth, batch_size=batch_size, randomize=False, device=self.device, dtype=self.dtype), desc="Rendering Frame")):
                batch_rgb_map = self.compiled_volume_render(batch_points, batch_depths, test_timestamp, poses, focals, width, height, near, far)
                rgb_map_list.append(batch_rgb_map.clone())
            rgb_map_flat = torch.cat(rgb_map_list, dim=0)  # (H * W, 3)
            rgb_map = rgb_map_flat.reshape(height[0].item(), width[0].item(), rgb_map_flat.shape[-1])  # (H, W, 3)
            rgb8 = (255 * np.clip(rgb_map.cpu().numpy(), 0, 1)).astype(np.uint8)  # (H, W, 3)
            imageio.imwrite(os.path.join(output_dir, 'rgb_{:03d}.png'.format(target_timestamp)), rgb8)

    def export_density_grid(self, save_ckp_path, resolution: int, target_timestamp: int, output_dir="output"):
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)

        ckpt = torch.load(save_ckp_path)
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.model.load_state_dict(ckpt['model_state_dict'])
        poses, focals, width, height, near, far = self.load_cameras_data(*camera_calibrations)
        focals = focals * self.ratio
        width = width * self.ratio
        height = height * self.ratio

        den = self.compiled_query_density_grid(x_min=scene_min_current[0], x_max=scene_max_current[0], y_min=scene_min_current[1], y_max=scene_max_current[1], z_min=scene_min_current[2], z_max=scene_max_current[2], res=resolution, time=float(target_timestamp / 120.0), poses=poses, focals=focals, width=width, height=height, near=near, far=far)
        np.savez_compressed(f"{output_dir}/density_{target_timestamp:03d}.npz", den=den.cpu().numpy())

    def query_rgb_map_skip_ghost(self, batch_input_xyzt_flat: torch.Tensor, batch_depths: torch.Tensor, in_all_frustum_mask: torch.Tensor):
        raw_flat = self.model(self.encoder(batch_input_xyzt_flat))  # (batch_size * depth, 4)
        in_all_frustum_mask_float = in_all_frustum_mask.unsqueeze(-1).float()  # (N,1)
        raw_flat = raw_flat * in_all_frustum_mask_float
        raw = raw_flat.reshape(-1, self.depth, 1)  # (batch_size, depth, 4)
        rgb_trained = torch.ones(3, device=self.device) * (0.6 + torch.tanh(self.model.rgb) * 0.4)
        alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1]) * batch_depths)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        batch_rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)
        return batch_rgb_map

    def load_videos_data(self, *video_paths, ratio: float):
        """
        Load multiple videos directly from given paths onto the specified device, resample images by ratio.

        Args:
        - *paths: str (arbitrary number of video file paths)

        Returns:
        - torch.Tensor of shape (T, V, H * ratio, W * ratio, C)
        """

        if not video_paths:
            raise ValueError("No video paths provided.")

        valid_paths = []
        for video_path in video_paths:
            _path = os.path.normpath(video_path)
            if not Path(_path).exists():
                raise FileNotFoundError(f"Video path {_path} does not exist.")
            valid_paths.append(_path)

        _frames_tensors = []
        for _path in valid_paths:
            try:
                _frames, _, _ = io.read_video(_path, pts_unit="sec")
                _frames = _frames.to(dtype=self.dtype) / 255.0
                _frames_tensors.append(_frames)
            except Exception as e:
                print(f"Error loading video '{_path}': {e}")

        videos = torch.stack(_frames_tensors)

        V, T, H, W, C = videos.shape
        videos_permuted = videos.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W)
        new_H, new_W = int(H * ratio), int(W * ratio)
        videos_resampled = torch.nn.functional.interpolate(videos_permuted, size=(new_H, new_W), mode='bilinear', align_corners=False)
        videos_resampled = videos_resampled.reshape(V, T, C, new_H, new_W).permute(1, 0, 3, 4, 2)

        return videos_resampled

    def load_cameras_data(self, *cameras_paths):
        """
        Load multiple camera calibration files directly from given paths onto the specified device.

        Args:
        - *cameras_paths: str (arbitrary number of camera calibration file paths)

        Returns:
        - poses: torch.Tensor of shape (N, 4, 4)
        - focals: torch.Tensor of shape (N)
        - width: torch.Tensor of shape (N)
        - height: torch.Tensor of shape (N)
        - near: torch.Tensor of shape (N)
        - far: torch.Tensor of shape (N)
        """

        if not cameras_paths:
            raise ValueError("No cameras paths provided.")

        valid_paths = []
        for camera_path in cameras_paths:
            _path = os.path.normpath(camera_path)
            if not Path(_path).exists():
                raise FileNotFoundError(f"Camera path {_path} does not exist.")
            valid_paths.append(_path)

        import numpy as np
        camera_infos = [np.load(path) for path in cameras_paths]
        widths = [int(info["width"]) for info in camera_infos]
        assert len(set(widths)) == 1, f"Error: Inconsistent widths found: {widths}. All cameras must have the same resolution."
        heights = [int(info["height"]) for info in camera_infos]
        assert len(set(heights)) == 1, f"Error: Inconsistent heights found: {heights}. All cameras must have the same resolution."
        nears = [float(info["near"]) for info in camera_infos]
        assert len(set(nears)) == 1, f"Error: Inconsistent nears found: {nears}. All cameras must have the same near plane."
        fars = [float(info["far"]) for info in camera_infos]
        assert len(set(fars)) == 1, f"Error: Inconsistent fars found: {fars}. All cameras must have the same far plane."
        poses = torch.stack([torch.tensor(info["cam_transform"], device=self.device, dtype=self.dtype) for info in camera_infos])
        focals = torch.tensor([info["focal"] * widths[0] / info["aperture"] for info in camera_infos], device=self.device, dtype=self.dtype)
        widths = torch.tensor(widths, device=self.device, dtype=torch.int32)
        heights = torch.tensor(heights, device=self.device, dtype=torch.int32)
        nears = torch.tensor(nears, device=self.device, dtype=self.dtype)
        fars = torch.tensor(fars, device=self.device, dtype=self.dtype)

        return poses, focals, widths, heights, nears, fars


def test_pipeline(rank, gpu_size):
    device = torch.device(f"cuda:{rank % gpu_size}")
    print(f"Process {rank} running on {device}")

    pipeline = PISGPipelineTorch(torch_device=device, torch_dtype=torch.float32)

    for _ in range(120):
        if _ % 2 == rank:
            pipeline.test(save_ckp_path="ckpt.tar", target_timestamp=_)


def export_density(rank, gpu_size):
    device = torch.device(f"cuda:{rank % gpu_size}")
    print(f"Process {rank} running on {device}")

    pipeline = PISGPipelineTorch(torch_device=device, torch_dtype=torch.float32)

    import tqdm
    for _ in tqdm.trange(120):
        if _ % 2 == rank:
            pipeline.export_density_grid(save_ckp_path="ckpt.tar", target_timestamp=_, resolution=128, output_dir="output/den")


def run_multidevice(func):
    gpu_size = torch.cuda.device_count()
    print(f"Launching {gpu_size} processes for GPU tasks.")
    mp.spawn(func, args=(gpu_size,), nprocs=gpu_size, join=True)


def train(target_device):
    pipeline = PISGPipelineTorch(torch_device=target_device, torch_dtype=torch.float32)
    pipeline.train(batch_size=1024, save_ckp_path="ckpt.tar")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    train(target_device=torch.device("cuda:0"))
    # run_multidevice(test_pipeline)
    # run_multidevice(export_density)
