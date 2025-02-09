from dataloaders.dataloader_hyfluid import (
    VideoInfos,
    CameraInfos,
    hyfluid_video_infos,
    hyfluid_camera_infos_list,
    load_videos_data_high_memory_numpy,
    load_videos_data_device,
    resample_images_by_ratio_numpy,
    resample_images_by_ratio_device
)
from utils.utils_nerf import (
    generate_rays_numpy,
    generate_rays_device,
    resample_images_scipy,
    resample_images_torch,
    get_points_device
)
from spatial_structures.occupancy_grid import points_in_camera_frustum
from model.model_hyfluid import NeRFSmall
from model.encoder_hyfluid import HashEncoderNative
import torch
import numpy as np
import tqdm
import random
import os

from dataclasses import dataclass
# from pytorch_memlab import profile_every
from torch.amp import autocast


@dataclass
class HyFluidArguments:
    total_iters: int = 10000
    batch_size: int = 256

    near: float = 1.1
    far: float = 1.5
    depth: int = 192
    ratio = 0.5

    encoder_num_scale: int = 16


args = HyFluidArguments()


class HyFluidPipeline:
    def __init__(self, video_infos: VideoInfos, camera_infos: list[CameraInfos], device, dtype_numpy, dtype_device):
        self.video_infos = video_infos
        self.camera_infos = camera_infos
        self.device = device
        self.dtype_numpy = dtype_numpy
        self.dtype_device = dtype_device

    def train_density_numpy(self, save_ckp_path=None):
        """
        Init data on cpu, train the model totally on the device. much slower than train_density_device
        """
        # 0. constants

        # 1. load encoder, model, optimizer
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=args.encoder_num_scale * 2).to(self.device)
        optimizer = torch.optim.RAdam([{'params': model_device.parameters(), 'weight_decay': 1e-6}, {'params': encoder_device.parameters(), 'eps': 1e-15}], lr=0.01, betas=(0.9, 0.99))

        # 2. load data to cpu
        train_video_data_numpy = load_videos_data_high_memory_numpy(self.video_infos, dataset_type="train", dtype=self.dtype_numpy)  # (#videos, #frames, H, W, C)
        train_video_data_numpy = resample_images_by_ratio_numpy(train_video_data_numpy, ratio=args.ratio)  # (#videos, #frames, H * ratio, W * ratio, C)
        train_video_data_numpy = train_video_data_numpy.transpose(1, 0, 2, 3, 4)  # (#frames, #videos, H * ratio, W * ratio, C)
        width, height, N_frames = train_video_data_numpy.shape[3], train_video_data_numpy.shape[2], train_video_data_numpy.shape[0]

        # 3. load poses
        train_indices = [0, 1, 2, 3]
        train_poses_numpy = np.array([self.camera_infos[i].transform_matrices for i in train_indices], dtype=self.dtype_numpy)  # (#cameras, 4, 4)
        focals_numpy = np.array([0.5 * width / np.tan(0.5 * np.array(self.camera_infos[i].camera_angle_x[0], dtype=self.dtype_numpy)) for i in train_indices], dtype=self.dtype_numpy)  # (#cameras)

        # 4. train
        N_rays = len(train_indices) * height * width
        rays_iter = N_rays

        rays_origin_flatten_numpy, rays_direction_flatten_numpy, rays_random_idxs_numpy, train_video_resampled_flatten_numpy = None, None, None, None
        for _ in tqdm.trange(0, args.total_iters):
            # resample rays
            if rays_iter >= N_rays:
                tqdm.tqdm.write(f"Resampling rays...")
                _rays_origin_numpy, _rays_direction_numpy, _u_numpy, _v_numpy = generate_rays_numpy(train_poses_numpy, focals=focals_numpy, width=width, height=height, randomize=True)  # (#cameras, H, W, 3), (H, W)
                rays_origin_flatten_numpy, rays_direction_flatten_numpy = _rays_origin_numpy.reshape(-1, 3), _rays_direction_numpy.reshape(-1, 3)  # (#cameras * H * W, 3), (#cameras * H * W, 3)
                rays_random_idxs_numpy = np.random.permutation(N_rays).astype(np.int32)  # (#cameras * H * W)
                train_video_resampled_flatten_numpy = resample_images_scipy(train_video_data_numpy, _u_numpy, _v_numpy).reshape(N_frames, -1, 3)  # (#frames, #cameras * H * W, 3)
                rays_iter = 0
            pixels_idxs = rays_random_idxs_numpy[rays_iter:rays_iter + args.batch_size]
            rays_iter += args.batch_size

            # get target frame (continuous)
            frame = random.uniform(0, N_frames - 1)
            frame_floor, frame_ceil, frames_alpha = int(frame), int(frame) + 1, frame - int(frame)
            target_frame_numpy = (1 - frames_alpha) * train_video_resampled_flatten_numpy[frame_floor] + frames_alpha * train_video_resampled_flatten_numpy[frame_ceil]  # (#cameras * H * W, 3)

            # get batch data
            batch_ray_origins = torch.tensor(rays_origin_flatten_numpy[pixels_idxs], device=self.device, dtype=self.dtype_device)  # (#batch, 3)
            batch_ray_directions = torch.tensor(rays_direction_flatten_numpy[pixels_idxs], device=self.device, dtype=self.dtype_device)  # (#batch, 3)
            batch_target_pixels = torch.tensor(target_frame_numpy[pixels_idxs], device=self.device, dtype=self.dtype_device)  # (#batch, 3)
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

    # @profile_every()
    def train_density_device(self, save_ckp_path=None):
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
        width, height, N_frames = train_video_data_device.shape[3], train_video_data_device.shape[2], train_video_data_device.shape[0]

        # 3. load poses
        train_indices = [0, 1, 2, 3]
        train_poses_device = torch.tensor([self.camera_infos[i].transform_matrices for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)
        focals_device = torch.tensor([0.5 * width / torch.tan(0.5 * torch.tensor(self.camera_infos[i].camera_angle_x[0], dtype=self.dtype_device)) for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras)

        # 4. train
        N_rays = len(train_indices) * height * width
        rays_iter = N_rays

        rays_origin_flatten_device, rays_direction_flatten_device, rays_random_idxs_device, train_video_resampled_flatten_device = None, None, None, None
        for _ in tqdm.trange(0, args.total_iters):
            # resample rays
            if rays_iter >= N_rays:
                tqdm.tqdm.write(f"Resampling rays...")
                _rays_origin_device, _rays_direction_device, _u_device, _v_device = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=True)  # (#cameras, H, W, 3), (H, W)
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

    def train_density_with_visualizer_device(self, save_ckp_path=None, resolution=64):
        """
        Train the model totally on the device, and save the grid_raw for visualization
        """
        # 0. constants
        os.makedirs("output", exist_ok=True)

        # 1. load encoder, model, optimizer
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=args.encoder_num_scale * 2).to(self.device)
        optimizer = torch.optim.RAdam([{'params': model_device.parameters(), 'weight_decay': 1e-6}, {'params': encoder_device.parameters(), 'eps': 1e-15}], lr=0.01, betas=(0.9, 0.99))

        # 2. load data to device
        train_video_data_device = load_videos_data_device(self.video_infos, dataset_type="train", device=self.device, dtype=self.dtype_device)  # (#videos, #frames, H, W, C)
        train_video_data_device = resample_images_by_ratio_device(train_video_data_device, ratio=args.ratio)  # (#videos, #frames, H * ratio, W * ratio, C)
        train_video_data_device = train_video_data_device.permute(1, 0, 2, 3, 4)  # (#frames, #videos, H * ratio, W * ratio, C)
        width, height, N_frames = train_video_data_device.shape[3], train_video_data_device.shape[2], train_video_data_device.shape[0]

        # 3. load poses
        train_indices = [0, 1, 2, 3]
        train_poses_device = torch.tensor([self.camera_infos[i].transform_matrices for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)
        focals_device = torch.tensor([0.5 * width / torch.tan(0.5 * torch.tensor(self.camera_infos[i].camera_angle_x[0], dtype=self.dtype_device)) for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras)

        # 4. train
        _xs, _ys, _zs = torch.meshgrid([torch.linspace(0, 1, resolution), torch.linspace(0, 1, resolution), torch.linspace(0, 1, resolution)], indexing='ij')
        _grids_device = torch.stack([_xs, _ys, -_zs], -1).to(self.device)
        _timesteps_device = torch.arange(N_frames, device=self.device, dtype=self.dtype_device) / (N_frames - 1)  # (N_frames)
        _timesteps_expanded_device = _timesteps_device.view(N_frames, 1, 1, 1, 1).expand(-1, resolution, resolution, resolution, 1)  # (N_frames, res, res, res, 1)
        _grids_expanded_device = _grids_device.unsqueeze(0).expand(N_frames, -1, -1, -1, -1)  # (N_frames, res, res, res, 3)
        grids_input_xyzt_device = torch.cat([_grids_expanded_device, _timesteps_expanded_device], dim=-1)  # (N_frames, res, res, res, 4)
        grids_input_xyzt_flat_device = grids_input_xyzt_device.reshape(-1, 4)  # (N_frames * res * res * res, 4)

        N_rays = len(train_indices) * height * width
        rays_iter = N_rays

        loss_history = []
        rays_origin_flatten_device, rays_direction_flatten_device, rays_random_idxs_device, train_video_resampled_flatten_device = None, None, None, None
        for _ in tqdm.trange(0, args.total_iters):
            # resample rays
            if rays_iter >= N_rays:
                tqdm.tqdm.write(f"Resampling rays...")
                _rays_origin_device, _rays_direction_device, _u_device, _v_device = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=True)  # (#cameras, H, W, 3), (H, W)
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
                loss_history.append(loss_image.item())

            if _ % 100 == 0:
                with torch.no_grad():
                    delta = N_frames
                    offset_points = resolution * resolution * resolution
                    grid_raw_flat_list = []
                    for h in range(delta):
                        chunk_test_input_xyzt_flat = grids_input_xyzt_flat_device[h * offset_points:(h + 1) * offset_points]
                        chunk_raw_flat = model_device(encoder_device(chunk_test_input_xyzt_flat))
                        grid_raw_flat_list.append(chunk_raw_flat)
                    grid_raw_flat = torch.cat(grid_raw_flat_list, 0)
                    grid_raw = grid_raw_flat.reshape(N_frames, resolution, resolution, resolution, 1)
                    np.savez_compressed(f"output/grid_raw_{_ // 100:03d}.npy", den=grid_raw.cpu().numpy())

        loss_array = np.array(loss_history)
        np.save("output/loss.npy", loss_array)

        if save_ckp_path is not None:
            torch.save({
                'encoder_state_dict': encoder_device.state_dict(),
                'model_state_dict': model_device.state_dict(),
                'width': width,
                'height': height,
                'N_frames': N_frames,
            }, save_ckp_path)

    def test_density_device(self, save_ckp_path=None, output_dir="output"):
        """
        Test the model totally on the device
        """
        # 0. constants
        import imageio.v3 as imageio
        os.makedirs(output_dir, exist_ok=True)

        # 1. load encoder, model, optimizer
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=args.encoder_num_scale * 2).to(self.device)
        ckpt = torch.load(save_ckp_path)
        encoder_device.load_state_dict(ckpt['encoder_state_dict'])
        model_device.load_state_dict(ckpt['model_state_dict'])
        width, height, N_frames = ckpt['width'], ckpt['height'], ckpt['N_frames']

        # 2. load poses
        train_indices = [4]
        train_poses_device = torch.tensor([self.camera_infos[i].transform_matrices for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)
        focals_device = torch.tensor([0.5 * width / torch.tan(0.5 * torch.tensor(self.camera_infos[i].camera_angle_x[0], dtype=self.dtype_device)) for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras)
        test_timesteps_device = torch.arange(N_frames, device=self.device, dtype=self.dtype_device) / (N_frames - 1)

        # 3. resample rays
        rays_origin_device, rays_direction_device, _, _ = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=False)  # (#cameras, H, W, 3), (#cameras, H, W, 3)
        rays_origin_device, rays_direction_device = rays_origin_device.reshape(rays_origin_device.shape[0], -1, 3), rays_direction_device.reshape(rays_direction_device.shape[0], -1, 3)  # (#cameras, H * W, 3), (#cameras, H * W, 3)

        with torch.no_grad():
            for rays_o, rays_d in zip(rays_origin_device, rays_direction_device):
                points, depths = get_points_device(rays_o, rays_d, args.near, args.far, args.depth, randomize=False)  # (H * W, #depth, 3), (H * W, #depth)
                points_flat = points.reshape(-1, 3)  # (H * W * #depth, 3)

                for _ in tqdm.trange(0, N_frames):
                    rgb_trained = torch.ones(3, device=self.device) * (0.6 + torch.tanh(model_device.rgb) * 0.4)

                    test_timesteps_expended = test_timesteps_device[_].expand(points_flat[..., :1].shape)  # (H * W * #depth, 1)
                    test_input_xyzt_flat = torch.cat([points_flat, test_timesteps_expended], dim=-1)  # (H * W * #depth, 4)

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

    def test_og(self):
        width, height = 1080, 1920
        near, far = 1.1, 1.5
        depth = 192

        train_indices = [4]
        train_poses_device = torch.tensor([self.camera_infos[i].transform_matrices for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)
        focals_device = torch.tensor([0.5 * width / torch.tan(0.5 * torch.tensor(self.camera_infos[i].camera_angle_x[0], dtype=self.dtype_device)) for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras)
        rays_origin_device, rays_direction_device, _, _ = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=False)  # (#cameras, H, W, 3), (#cameras, H, W, 3)
        rays_origin_device, rays_direction_device = rays_origin_device.reshape(rays_origin_device.shape[0], -1, 3), rays_direction_device.reshape(rays_direction_device.shape[0], -1, 3)  # (#cameras, H * W, 3), (#cameras, H * W, 3)
        rays_o, rays_d = rays_origin_device[0], rays_direction_device[0]
        points, _ = get_points_device(rays_o, rays_d, 1.1, 1.5, depth, randomize=False)
        camera_angle_x = torch.tensor([self.camera_infos[i].camera_angle_x[0] for i in train_indices], device=self.device, dtype=self.dtype_device)

        visible_mask, frustum_corners_world = points_in_camera_frustum(
            points.reshape(-1, 3),
            train_poses_device[0],
            camera_angle_x[0].item(),
            width,
            height,
            near,
            far
        )

        points = points.reshape(width, height, depth, 3)
        visible_mask = visible_mask.reshape(width, height, depth)

        # 计算 True 的数量
        true_count = visible_mask.sum()

        # 计算 False 的数量
        false_count = visible_mask.numel() - true_count  # 总元素数减去 True 的数量

        print(f"Number of True: {true_count}")
        print(f"Number of False: {false_count}")

        np.savez("points", points=points[::1000].cpu().numpy(), points_mask=points[visible_mask][::100].cpu().numpy())


if __name__ == '__main__':
    hyfluid_video_infos.root_dir = "../data/hyfluid"
    hyfluid = HyFluidPipeline(hyfluid_video_infos, hyfluid_camera_infos_list, device=torch.device("cuda"), dtype_numpy=np.float32, dtype_device=torch.float32)
    # hyfluid.train_density_numpy("final_ckp.tar")
    # hyfluid.train_density_device("final_ckp.tar")
    # hyfluid.test_density_device("final_ckp.tar")

    hyfluid.test_og()
