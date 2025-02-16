from dataloaders.dataloader_hyfluid import (
    VideoInfos,
    load_videos_data_device,
    resample_images_by_ratio_device
)
from utils.utils_nerf import (
    generate_rays_device,
    resample_images_torch,
    get_points_device
)
from model.model_hyfluid import NeRFSmall
from model.encoder_hyfluid import HashEncoderNative

import torch
import numpy as np
import tqdm
import random
import os
import math

from dataclasses import dataclass
from pathlib import Path
from torch.amp import autocast


@dataclass
class PISGArguments:
    total_iters: int = 10000
    batch_size: int = 256

    near: float = 10
    far: float = 21.6
    depth: int = 192

    encoder_num_scale: int = 16
    ratio = 1.0


args = PISGArguments()

infos = VideoInfos(
    root_dir=Path("../data/PISG/scene1"),
    train_videos=[Path("back.mp4"), Path("front.mp4"), Path("right.mp4"), Path("top.mp4")],
    validation_videos=[],
    test_videos=[],
)


class PISGPipeline:
    def __init__(self, video_infos: VideoInfos, device, dtype_numpy, dtype_device):
        self.video_infos = video_infos
        self.device = device
        self.dtype_numpy = dtype_numpy
        self.dtype_device = dtype_device

    def train_device(self, save_ckp_path=None):
        """
        Train the model totally on the device, much faster than train_density_numpy
        """
        # 0. constants

        # 1. load encoder, model, optimizer
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=args.encoder_num_scale * 2).to(self.device)
        optimizer = torch.optim.RAdam([{'params': model_device.parameters(), 'weight_decay': 1e-6}, {'params': encoder_device.parameters(), 'eps': 1e-15}], lr=0.01, betas=(0.9, 0.99))

        # 新增：使用渐进式指数衰减，设定训练结束时的学习率为初始学习率的 10%
        target_lr_ratio = 0.1  # 你可以根据需求调整目标比例
        gamma = math.exp(math.log(target_lr_ratio) / args.total_iters)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # 2. load data to device
        train_video_data_device = load_videos_data_device(self.video_infos, dataset_type="train", device=self.device, dtype=self.dtype_device)  # (#videos, #frames, H, W, C)
        train_video_data_device = resample_images_by_ratio_device(train_video_data_device, ratio=args.ratio)  # (#videos, #frames, H * ratio, W * ratio, C)
        train_video_data_device = train_video_data_device.permute(1, 0, 2, 3, 4)  # (#frames, #videos, H * ratio, W * ratio, C)
        width, height, N_videos, N_frames = train_video_data_device.shape[3], train_video_data_device.shape[2], train_video_data_device.shape[1], train_video_data_device.shape[0]

        # 3. load poses
        camera_infos_path = [
            Path("cam_back.npz"),
            Path("cam_front.npz"),
            Path("cam_right.npz"),
            Path("cam_top.npz"),
        ]
        camera_infos = [np.load(infos.root_dir / path) for path in camera_infos_path]
        cam_transforms = [torch.tensor(info["cam_transform"], device=self.device, dtype=torch.float32) for info in camera_infos]
        train_poses_device = torch.stack(cam_transforms)
        focal_pixels_device = torch.tensor([info["focal"] * width / info["aperture"] for info in camera_infos], device=self.device, dtype=torch.float32)

        # 4. train
        N_rays = N_videos * height * width
        rays_iter = N_rays

        loss_avg_list = []
        loss_accum = 0.0

        rays_origin_flatten_device, rays_direction_flatten_device, rays_random_idxs_device, train_video_resampled_flatten_device = None, None, None, None
        for _ in tqdm.trange(0, args.total_iters):
            # resample rays
            if rays_iter >= N_rays:
                tqdm.tqdm.write(f"Resampling rays...")
                _rays_origin_device, _rays_direction_device, _u_device, _v_device = generate_rays_device(train_poses_device, focals=focal_pixels_device, width=width, height=height, randomize=True)  # (#cameras, H, W, 3), (H, W)
                rays_origin_flatten_device, rays_direction_flatten_device = _rays_origin_device.reshape(-1, 3), _rays_direction_device.reshape(-1, 3)  # (#cameras * H * W, 3), (#cameras * H * W, 3)
                rays_random_idxs_device = torch.randperm(N_rays, device=self.device, dtype=torch.int32)  # (#cameras * H * W)
                train_video_resampled_flatten_device = resample_images_torch(train_video_data_device, _u_device, _v_device).reshape(N_frames, -1, 3)  # (#frames, #cameras * H * W, 3)
                rays_iter = 0
            pixels_idxs = rays_random_idxs_device[rays_iter:rays_iter + args.batch_size]
            rays_iter += args.batch_size

            # get target frame (continuous)
            frame = random.uniform(0, N_frames - 1)
            frame_floor, frame_ceil, frames_alpha = int(frame), int(frame) + 1, frame - int(frame)
            target_frame_device = (1 - frames_alpha) * train_video_resampled_flatten_device[frame_floor] + frames_alpha * train_video_resampled_flatten_device[frame_ceil]  # (#cameras * H * W, 3)

            # get batch data
            batch_ray_origins = rays_origin_flatten_device[pixels_idxs]  # (#batch, 3)
            batch_ray_directions = rays_direction_flatten_device[pixels_idxs]  # (#batch, 3)
            batch_target_pixels = target_frame_device[pixels_idxs]  # (#batch, 3)
            batch_points, batch_depths = get_points_device(batch_ray_origins, batch_ray_directions, args.near, args.far, args.depth, randomize=True)  # (#batch, #depth, 3), (#batch, #depth)

            # ===== 新增：归一化 xyz 坐标 =====
            # 假设场景坐标范围为 [-1, 1]，归一化到 [0, 1] 内。如果你的场景包围盒不同，请相应调整 scene_min 和 scene_max
            scene_min = torch.tensor([-20.0, -20.0, -20.0], device=self.device, dtype=batch_points.dtype)
            scene_max = torch.tensor([20.0, 20.0, 20.0], device=self.device, dtype=batch_points.dtype)
            batch_points_normalized = (batch_points - scene_min) / (scene_max - scene_min)
            # ===============================

            batch_time = torch.tensor(frame / (N_frames - 1), device=self.device, dtype=self.dtype_device)
            batch_input_xyzt = torch.cat([batch_points_normalized, batch_time.expand(batch_points_normalized[..., :1].shape)], dim=-1)  # (#batch, #depth , 4)
            batch_input_xyzt_flat = batch_input_xyzt.reshape(-1, 4)  # (#batch * #depth, 4)

            # forward
            raw_flat = model_device(encoder_device(batch_input_xyzt_flat))  # (#batch * #depth, 1)
            raw = raw_flat.reshape(-1, args.depth, 1)  # (#batch, #depth, 1)
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

            # Added: 累计loss并每100次iter记录平均loss
            loss_accum += loss_image.item()
            if _ % 100 == 0 and _ > 0:
                loss_avg = loss_accum / 100.0
                loss_avg_list.append(loss_avg)
                loss_accum = 0.0
                tqdm.tqdm.write(f"Average loss over iterations {_ - 98} to {_ + 1}: {loss_avg}")

        if save_ckp_path is not None:
            torch.save({
                'encoder_state_dict': encoder_device.state_dict(),
                'model_state_dict': model_device.state_dict(),
                'width': width,
                'height': height,
                'N_frames': N_frames,
            }, save_ckp_path)

        # Added: 绘制每100次iter记录的loss平均值图
        import matplotlib.pyplot as plt
        plt.figure()
        iterations = list(range(100, args.total_iters + 1, 100))
        plt.plot(iterations, loss_avg_list, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss")
        plt.title("Average Loss per 100 Iterations")
        plt.grid(True)
        plt.show()

    def test_density(self, save_ckp_path=None, output_dir="output"):
        """
        Test the model totally on the device
        """
        # 0. constants
        import imageio.v3 as imageio
        os.makedirs(output_dir, exist_ok=True)

        # 1. load encoder, model
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=args.encoder_num_scale * 2).to(self.device)
        ckpt = torch.load(save_ckp_path)
        encoder_device.load_state_dict(ckpt['encoder_state_dict'])
        model_device.load_state_dict(ckpt['model_state_dict'])
        width, height, N_frames = ckpt['width'], ckpt['height'], ckpt['N_frames']

        # 2. load poses
        camera_infos_path = [
            Path("cam_front.npz"),
        ]
        camera_infos = [np.load(infos.root_dir / path) for path in camera_infos_path]
        cam_transforms = [torch.tensor(info["cam_transform"], device=self.device, dtype=torch.float32) for info in camera_infos]
        train_poses_device = torch.stack(cam_transforms)
        focals_device = torch.tensor([info["focal"] * width / info["aperture"] for info in camera_infos], device=self.device, dtype=torch.float32)

        test_timesteps_device = torch.arange(N_frames, device=self.device, dtype=self.dtype_device) / (N_frames - 1)

        # 3. resample rays
        rays_origin_device, rays_direction_device, _, _ = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=False)  # (#cameras, H, W, 3), (#cameras, H, W, 3)
        rays_origin_device, rays_direction_device = rays_origin_device.reshape(rays_origin_device.shape[0], -1, 3), rays_direction_device.reshape(rays_direction_device.shape[0], -1, 3)  # (#cameras, H * W, 3), (#cameras, H * W, 3)

        with torch.no_grad():
            for rays_o, rays_d in zip(rays_origin_device, rays_direction_device):
                points, depths = get_points_device(rays_o, rays_d, args.near, args.far, args.depth, randomize=False)  # (H * W, #depth, 3), (H * W, #depth)
                points_flat = points.reshape(-1, 3)  # (H * W * #depth, 3)

                # ===== 新增：归一化 xyz 坐标 =====
                # 假设场景坐标范围为 [-1, 1]，归一化到 [0, 1] 内
                scene_min = torch.tensor([-20.0, -20.0, -20.0], device=self.device, dtype=points_flat.dtype)
                scene_max = torch.tensor([20.0, 20.0, 20.0], device=self.device, dtype=points_flat.dtype)
                points_normalized = (points_flat - scene_min) / (scene_max - scene_min)
                # ===============================

                for _ in tqdm.trange(0, N_frames):
                    rgb_trained = torch.ones(3, device=self.device) * (0.6 + torch.tanh(model_device.rgb) * 0.4)

                    test_timesteps_expended = test_timesteps_device[_].expand(points_normalized[..., :1].shape)  # (H * W * #depth, 1)
                    test_input_xyzt_flat = torch.cat([points_normalized, test_timesteps_expended], dim=-1)  # (H * W * #depth, 4)

                    with autocast("cuda"):
                        rgb_map_flat_list = []
                        ratio = 1
                        delta = height // ratio
                        offset_points = width * args.depth * ratio
                        offset_depths = width * ratio
                        for h in range(delta):
                            chunk_test_input_xyzt_flat = test_input_xyzt_flat[h * offset_points:(h + 1) * offset_points]  # (chuck * #depth, 4)
                            chunk_depths = depths[h * offset_depths:(h + 1) * offset_depths]  # (chuck, #depth)
                            chunk_raw_flat = model_device(encoder_device(chunk_test_input_xyzt_flat))  # (chuck * #depth, 1)
                            chunk_raw = chunk_raw_flat.reshape(-1, args.depth, 1)  # (chuck, #depth, 1)
                            chunk_alpha = 1. - torch.exp(-torch.nn.functional.relu(chunk_raw[..., -1]) * chunk_depths)  # (chuck, #depth)
                            chunk_weights = chunk_alpha * torch.cumprod(torch.cat([torch.ones((chunk_alpha.shape[0], 1), device=self.device), 1. - chunk_alpha + 1e-10], -1), -1)[:, :-1]  # (chuck, #depth)
                            chunk_rgb_map_flat = torch.sum(chunk_weights[..., None] * rgb_trained, -2)  # (chuck, 3)
                            rgb_map_flat_list.append(chunk_rgb_map_flat)  # (chuck, 3)
                        rgb_map_flat = torch.cat(rgb_map_flat_list, 0)  # (H * W, 3)
                    rgb_map = rgb_map_flat.reshape(height, width, rgb_map_flat.shape[-1])  # (H, W, 3)

                    rgb8 = (255 * np.clip(rgb_map.cpu().numpy(), 0, 1)).astype(np.uint8)  # (H, W, 3)
                    imageio.imwrite(os.path.join(output_dir, 'rgb_{:03d}.png'.format(_)), rgb8)


if __name__ == '__main__':
    target_device = torch.device("cuda")

    PISG = PISGPipeline(video_infos=infos, device=target_device, dtype_numpy=np.float32, dtype_device=torch.float32)
    PISG.train_device(save_ckp_path="final_ckp.tar")
    # PISG.test_density(save_ckp_path="final_ckp.tar", out1put_dir="output_PISG")
