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

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PISGArguments:
    total_iters: int = 30000
    batch_size: int = 256

    near: float = 10
    far: float = 21.6
    depth: int = 192

    encoder_num_scale: int = 16
    ratio = 0.5


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
        print(train_poses_device)
        print(focal_pixels_device)

        # 4. train
        N_rays = N_videos * height * width
        rays_iter = N_rays

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
            batch_time = torch.tensor(frame / (N_frames - 1), device=self.device, dtype=self.dtype_device)
            batch_input_xyzt = torch.cat([batch_points, batch_time.expand(batch_points[..., :1].shape)], dim=-1)  # (#batch, #depth , 4)
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
            if _ % 100 == 0:
                tqdm.tqdm.write(f"loss_image: {loss_image.item()}")

        if save_ckp_path is not None:
            torch.save({
                'encoder_state_dict': encoder_device.state_dict(),
                'model_state_dict': model_device.state_dict(),
                'width': width,
                'height': height,
                'N_frames': N_frames,
            }, save_ckp_path)


if __name__ == '__main__':
    target_device = torch.device("cuda")

    PISG = PISGPipeline(video_infos=infos, device=target_device, dtype_numpy=np.float32, dtype_device=torch.float32)
    PISG.train_device(save_ckp_path="final_ckp.tar")
