import torch
import torchvision.io as io
import os
import math
import random
from pathlib import Path

from model.model_hyfluid import NeRFSmall
from model.encoder_hyfluid import HashEncoderNative

from dataclasses import dataclass


@dataclass
class PISGArguments:
    total_iters: int = 30000
    batch_size: int = 256

    near: float = 10
    far: float = 21.6
    depth: int = 192

    encoder_num_scale: int = 16
    ratio = 1.0


args = PISGArguments()


def find_relative_paths(relative_path_list):
    current_dir = Path.cwd()
    search_dirs = [current_dir, current_dir.parent, current_dir.parent.parent]

    for i in range(len(relative_path_list)):
        found = False
        relative_path = relative_path_list[i]
        for directory in search_dirs:
            full_path = directory / relative_path
            if full_path.exists():
                relative_path_list[i] = str(full_path.resolve())  # 直接修改列表中的元素
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


class PISGPipelineTorch:
    def __init__(self, torch_device: torch.device, torch_dtype: torch.dtype):
        self.device = torch_device
        self.dtype = torch_dtype

    def train(self):
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=args.encoder_num_scale * 2).to(self.device)
        optimizer = torch.optim.RAdam([{'params': model_device.parameters(), 'weight_decay': 1e-6}, {'params': encoder_device.parameters(), 'eps': 1e-15}], lr=0.01, betas=(0.9, 0.99))

        # 新增：使用渐进式指数衰减，设定训练结束时的学习率为初始学习率的 10%
        target_lr_ratio = 0.1  # 你可以根据需求调整目标比例
        gamma = math.exp(math.log(target_lr_ratio) / args.total_iters)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        videos_data = self.load_videos_data(*training_videos).permute(1, 0, 2, 3, 4)  # (T, V, H, W, C)
        poses, focals, width, height, near, far = self.load_cameras_data(*camera_calibrations)

        import tqdm
        iter = 0
        loss_avg_list = []
        loss_accum = 0.0
        batch_size = 1024
        for _1 in tqdm.trange(0, 10):
            u, v, dirs = self.shuffle_uv(focals=focals, width=width, height=height, randomize=True)
            videos_data_resampled = self.resample_frames(frames=videos_data, u=u, v=v)  # (T, V, H, W, C)

            for _2, (batch_points, batch_depths, batch_indices) in enumerate(self.sample_frustum(dirs=dirs, poses=poses, near=near, far=far, depth=32, batch_size=batch_size, randomize=True)):
                iter += 1

                frame = random.uniform(0, videos_data_resampled.shape[0] - 1)
                frame_floor, frame_ceil, frames_alpha = int(frame), int(frame) + 1, frame - int(frame)
                target_frame = (1 - frames_alpha) * videos_data_resampled[frame_floor] + frames_alpha * videos_data_resampled[frame_ceil]  # (V * H * W, C)
                batch_target_pixels = target_frame[batch_indices]  # (batch_size, C)

                # ===== 新增：归一化 xyz 坐标 =====
                # 假设场景坐标范围为 [-1, 1]，归一化到 [0, 1] 内。如果你的场景包围盒不同，请相应调整 scene_min 和 scene_max
                scene_min = torch.tensor([-20.0, -20.0, -20.0], device=self.device, dtype=self.dtype)
                scene_max = torch.tensor([20.0, 20.0, 20.0], device=self.device, dtype=self.dtype)
                batch_points_normalized = (batch_points - scene_min) / (scene_max - scene_min)  # (batch_size, depth, 3)
                # ===============================

                batch_time = torch.tensor(frame / (videos_data_resampled.shape[0] - 1), device=self.device, dtype=self.dtype)
                batch_input_xyzt = torch.cat([batch_points_normalized, batch_time.expand(batch_points_normalized[..., :1].shape)], dim=-1)  # (batch_size, depth, 4)
                batch_input_xyzt_flat = batch_input_xyzt.reshape(-1, 4)  # (batch_size * depth, 4)

                # forward
                raw_flat = model_device(encoder_device(batch_input_xyzt_flat))  # (batch_size * depth, 4)
                raw = raw_flat.reshape(-1, args.depth, 1)  # (batch_size, depth, 4)
                rgb_trained = torch.ones(3, device=self.device) * (0.6 + torch.tanh(model_device.rgb) * 0.4)
                alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1]) * batch_depths)
                weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
                rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)

                # optimize loss
                loss_image = torch.nn.functional.mse_loss(rgb_map, batch_target_pixels)
                optimizer.zero_grad()
                loss_image.backward()
                optimizer.step()

                # 每个 iteration 后更新一次学习率
                scheduler.step()

                loss_accum += loss_image.item()
                if iter % 100 == 0 and iter > 0:
                    loss_avg = loss_accum / 100.0
                    loss_avg_list.append(loss_avg)
                    loss_accum = 0.0
                    tqdm.tqdm.write(f"Average loss over iterations {_ - 99} to {_}: {loss_avg}")

        save_ckp_path = "final_ckp.tar"
        if save_ckp_path is not None:
            torch.save({
                'encoder_state_dict': encoder_device.state_dict(),
                'model_state_dict': model_device.state_dict(),
                'width': width,
                'height': height,
                'N_frames': 120,
            }, save_ckp_path)

    def load_videos_data(self, *video_paths):
        """
        Load multiple videos directly from given paths onto the specified device.

        Args:
        - *paths: str (arbitrary number of video file paths)

        Returns:
        - torch.Tensor of shape (V, T, H, W, C)
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
                _frames = _frames.to(self.device, dtype=self.dtype) / 255.0
                _frames_tensors.append(_frames)
            except Exception as e:
                print(f"Error loading video '{_path}': {e}")

        return torch.stack(_frames_tensors)

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

    def resample_frames(self, frames: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
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

    def shuffle_uv(self, focals: torch.Tensor, width: int, height: int, randomize: bool):
        """
        Shuffle the UV coordinates of the image plane.

        Args:
        - focals: torch.Tensor of shape (N)
        - width: int
        - height: int
        - randomize: bool

        Returns:
        - u_normalized: torch.Tensor of shape (N, H, W)
        - v_normalized: torch.Tensor of shape (N, H, W)
        """
        u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=self.device, dtype=self.dtype), torch.linspace(0, height - 1, height, device=self.device, dtype=self.dtype), indexing='xy')  # (H, W), (H, W)
        if randomize:
            du, dv = torch.rand_like(u), torch.rand_like(v)  # (H, W), (H, W)
            u, v = torch.clip(u + du, 0, width - 1), torch.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
        u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)
        dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)

        return u, v, dirs

    def sample_frustum(self, dirs: torch.Tensor, poses: torch.Tensor, near: float, far: float, depth: int, batch_size: int, randomize: bool):
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

        depths = torch.linspace(near, far, steps=depth, device=self.device, dtype=self.dtype).unsqueeze(0)  # (1, depth)

        indices = torch.randperm(num_rays, device=self.device)  # (N*H*W)
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


if __name__ == '__main__':
    pipeline = PISGPipelineTorch(torch_device=torch.device("cuda"), torch_dtype=torch.float32)
    pipeline.train()
