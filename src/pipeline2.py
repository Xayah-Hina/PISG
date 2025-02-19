import torch
import torchvision.io as io
import torch.multiprocessing as mp
import os
import math
import random
from pathlib import Path

from model.model_hyfluid import NeRFSmall
from model.encoder_hyfluid import HashEncoderNative


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


training_videos = [
    "data/PISG/scene1/front.mp4",
    "data/PISG/scene1/right.mp4",
    "data/PISG/scene1/back.mp4",
    "data/PISG/scene1/top.mp4",
]

camera_calibrations = [
    "data/PISG/scene1/cam_front.npz",
    "data/PISG/scene1/cam_right.npz",
    "data/PISG/scene1/cam_back.npz",
    "data/PISG/scene1/cam_top.npz",
]

find_relative_paths(training_videos)
find_relative_paths(camera_calibrations)


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)

        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


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
    frame = random.uniform(0, videos_data.shape[0] - 1)
    frame_floor, frame_ceil, frames_alpha = int(frame), int(frame) + 1, frame - int(frame)
    target_frame = (1 - frames_alpha) * videos_data[frame_floor] + frames_alpha * videos_data[frame_ceil]  # (V * H * W, C)
    target_frame = target_frame.reshape(-1, 3)
    batch_target_pixels = target_frame[batch_indices]  # (batch_size, C)
    batch_time = torch.tensor(frame / (videos_data.shape[0] - 1), device=device, dtype=dtype)

    return batch_time, batch_target_pixels


def normalize_points(points: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """
    Normalize the points to the range [0, 1].

    Args:
    - points: torch.Tensor of shape (..., 3)

    Returns:
    - points_normalized: torch.Tensor of shape (..., 3)
    """

    scene_min = torch.tensor([-20.0, -20.0, -20.0], device=device, dtype=dtype)
    scene_max = torch.tensor([20.0, 20.0, 20.0], device=device, dtype=dtype)
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
    u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=device, dtype=dtype), torch.linspace(0, height - 1, height, device=device, dtype=dtype), indexing='xy')  # (H, W), (H, W)
    if randomize:
        du, dv = torch.rand_like(u), torch.rand_like(v)  # (H, W), (H, W)
        u, v = torch.clip(u + du, 0, width - 1), torch.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
    u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)
    dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)

    return dirs, u, v


class PISGPipelineVelTorch:
    """
    Pipeline for training and testing the PISG model using PyTorch.
    """

    def __init__(self, torch_device: torch.device, torch_dtype: torch.dtype):
        self.device = torch_device
        self.dtype = torch_dtype
        self.encoder_num_scale = 16
        self.depth = 192
        self.ratio = 1.0

        self.encoder = HashEncoderNative(device=self.device).to(self.device)
        self.model = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_num_scale * 2).to(self.device)
        self.optimizer = torch.optim.RAdam([{'params': self.model.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

        target_lr_ratio = 0.1  # 新增：使用渐进式指数衰减，设定训练结束时的学习率为初始学习率的 10% 你可以根据需求调整目标比例
        gamma = math.exp(math.log(target_lr_ratio) / 30000)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        def PISG_forward(batch_points: torch.Tensor, batch_depths: torch.Tensor, batch_time: torch.Tensor):
            batch_points_normalized = normalize_points(points=batch_points, device=self.device, dtype=self.dtype)  # (batch_size, depth, 3)
            batch_input_xyzt_flat = torch.cat([batch_points_normalized, batch_time.expand(batch_points_normalized[..., :1].shape)], dim=-1).reshape(-1, 4)  # (batch_size * depth, 4)
            batch_rgb_map = self.query_rgb_map(xyzt=batch_input_xyzt_flat, batch_depths=batch_depths)
            return batch_rgb_map

        self.compiled_forward = torch.compile(PISG_forward, mode="max-autotune")

    def calculate_jacobian(self, xyzt: torch.Tensor, chunk_size: int):

        jac_list = []
        for i in range(0, xyzt.shape[0], chunk_size):
            xyzt_chunk = xyzt[i:i + chunk_size]
            xyzt_chunk.requires_grad = True

            def g(x):
                return self.model(x)

            h = self.encoder(xyzt_chunk)
            jac = torch.vmap(torch.func.jacrev(g))(h)
            jac_x = _get_minibatch_jacobian(h, xyzt_chunk)
            jac = jac @ jac_x
            # _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]
            jac_list.append(jac)
        jac_final = torch.cat(jac_list, dim=0)

        return jac_final

    def train(self, batch_size, save_ckp_path):
        videos_data = self.load_videos_data(*training_videos, ratio=self.ratio)
        poses, focals, width, height, near, far = self.load_cameras_data(*camera_calibrations)
        width = int(width * self.ratio)
        height = int(height * self.ratio)

        import tqdm
        for _1 in tqdm.trange(0, 1):
            dirs, u, v = shuffle_uv(focals=focals, width=width, height=height, randomize=True, device=self.device, dtype=self.dtype)
            videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v)  # (T, V, H, W, C)

            for _2, (batch_points, batch_depths, batch_indices) in enumerate(
                    tqdm.tqdm(sample_frustum(dirs=dirs, poses=poses, near=near, far=far, depth=self.depth, batch_size=batch_size, randomize=True, device=self.device, dtype=self.dtype), desc="Frustum Sampling")):
                batch_time, batch_target_pixels = sample_random_frame(videos_data=videos_data_resampled, batch_indices=batch_indices, device=self.device, dtype=self.dtype)  # (batch_size, C)
                batch_rgb_map = self.compiled_forward(batch_points, batch_depths, batch_time)

                batch_points_normalized = normalize_points(points=batch_points, device=self.device, dtype=self.dtype)  # (batch_size, depth, 3)
                batch_input_xyzt_flat = torch.cat([batch_points_normalized, batch_time.expand(batch_points_normalized[..., :1].shape)], dim=-1).reshape(-1, 4)  # (batch_size * depth, 4)
                jac_final = self.calculate_jacobian(xyzt=batch_input_xyzt_flat, chunk_size=1024 * 64)
                print(jac_final.shape)

                loss_image = torch.nn.functional.mse_loss(batch_rgb_map, batch_target_pixels)
                self.optimizer.zero_grad()
                loss_image.backward()
                self.optimizer.step()
                self.scheduler.step()

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
        poses = poses[:1]
        focals = focals[:1]

        batch_size = 1024
        test_timestamp = torch.tensor(target_timestamp / 120., device=self.device, dtype=self.dtype)
        import tqdm
        with torch.no_grad():
            dirs, _, _ = shuffle_uv(focals=focals, width=width, height=height, randomize=False, device=self.device, dtype=self.dtype)
            rgb_map_list = []
            for _1, (batch_points, batch_depths, batch_indices) in enumerate(
                    tqdm.tqdm(sample_frustum(dirs=dirs, poses=poses, near=near, far=far, depth=self.depth, batch_size=batch_size, randomize=False, device=self.device, dtype=self.dtype), desc="Rendering Frame")):
                batch_rgb_map = self.compiled_forward(batch_points, batch_depths, test_timestamp)
                rgb_map_list.append(batch_rgb_map.clone())
            rgb_map_flat = torch.cat(rgb_map_list, dim=0)  # (H * W, 3)
            rgb_map = rgb_map_flat.reshape(height, width, rgb_map_flat.shape[-1])  # (H, W, 3)
            rgb8 = (255 * np.clip(rgb_map.cpu().numpy(), 0, 1)).astype(np.uint8)  # (H, W, 3)
            imageio.imwrite(os.path.join(output_dir, 'rgb_{:03d}.png'.format(target_timestamp)), rgb8)

    def query_rgb_map(self, xyzt: torch.Tensor, batch_depths: torch.Tensor):
        raw_flat = self.model(self.encoder(xyzt))  # (batch_size * depth, 4)
        raw = raw_flat.reshape(-1, self.depth, 1)  # (batch_size, depth, 4)
        rgb_trained = torch.ones(3, device=self.device) * (0.6 + torch.tanh(self.model.rgb) * 0.4)
        alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1]) * batch_depths)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)

        return rgb_map

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
        videos_resampled = videos_resampled.to(self.device)

        return videos_resampled

    def load_cameras_data(self, *cameras_paths):
        """
        Load multiple camera calibration files directly from given paths onto the specified device.

        Args:
        - *cameras_paths: str (arbitrary number of camera calibration file paths)

        Returns:
        - poses: torch.Tensor of shape (N, 4, 4)
        - focals: torch.Tensor of shape (N)
        - width: int
        - height: int
        - near: float
        - far: float
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

        return poses, focals, widths[0], heights[0], nears[0], fars[0]


def test_pipeline(rank, gpu_size):
    device = torch.device(f"cuda:{rank % gpu_size}")
    print(f"Process {rank} running on {device}")

    pipeline = PISGPipelineVelTorch(torch_device=device, torch_dtype=torch.float32)

    for _ in range(120):
        if _ % 2 == rank:
            pipeline.test(save_ckp_path="ckpt.tar", target_timestamp=_)


def test():
    gpu_size = torch.cuda.device_count()
    print(f"Launching {gpu_size} processes for GPU tasks.")
    mp.spawn(test_pipeline, args=(gpu_size,), nprocs=gpu_size, join=True)


def train():
    pipeline = PISGPipelineVelTorch(torch_device=torch.device("cuda"), torch_dtype=torch.float32)
    pipeline.train(batch_size=1024, save_ckp_path="ckpt.tar")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    train()
    # test()

# def test_jac(chunk_size):
#     import time
#     # 初始化 pipeline
#     pipeline = PISGPipelineVelTorch(torch_device=torch.device("cuda"), torch_dtype=torch.float32)
#
#     # 清除 CUDA 内存缓存，确保数据独立
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#
#     # 创建输入张量
#     input_tensor = torch.rand(256 * 192, 4, device=torch.device("cuda"), dtype=torch.float32)
#
#     # 记录开始时间
#     start_time = time.perf_counter()
#
#     # 计算 Jacobian
#     jac_final = pipeline.calculate_jacobian(input_tensor, chunk_size)
#
#     # 确保 GPU 计算完成
#     torch.cuda.synchronize()
#
#     # 记录结束时间
#     end_time = time.perf_counter()
#
#     # 获取最高显存占用（单位 MB）
#     max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
#     elapsed_time = end_time - start_time
#
#     print(
#         f"Chunk size: {chunk_size}, Jacobian shape: {jac_final.shape}, Max GPU Memory: {max_memory:.2f} MB, Time: {elapsed_time:.4f} sec")
#
#     # 释放 GPU 变量
#     del pipeline, input_tensor, jac_final
#
#     # 清空显存缓存，释放未使用的显存
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
#
#
# if __name__ == '__main__':
#     torch.set_float32_matmul_precision('high')
#     for cs in [512, 1024, 256 * 192]:
#         test_jac(cs)

# Chunk size: 512, Jacobian shape: torch.Size([49152, 1, 4]), Max GPU Memory: 12612.29 MB, Time: 7.2823 sec
# Chunk size: 1024, Jacobian shape: torch.Size([49152, 1, 4]), Max GPU Memory: 14310.39 MB, Time: 5.0719 sec
# Chunk size: 49152, Jacobian shape: torch.Size([49152, 1, 4]), Max GPU Memory: 39880.85 MB, Time: 9.3621 sec
