from dataloaders.dataloader_hyfluid import VideoInfos, CameraInfos, hyfluid_video_infos, hyfluid_camera_infos_list, load_videos_data_device
from utils.utils_nerf import generate_rays_device, resample_images_torch
from model.model_hyfluid import NeRFSmall, NeRFSmallPotential
from model.encoder_hyfluid import HashEncoderNative
import torch
import numpy as np


class HyFluidPipeline:
    def __init__(self, video_infos: VideoInfos, camera_infos: list[CameraInfos], device, dtype_numpy, dtype_device):
        self.video_infos = video_infos
        self.camera_infos = camera_infos
        self.device = device
        self.dtype_numpy = dtype_numpy
        self.dtype_device = dtype_device

    def train_density_device(self):
        """
        Train the model totally on the device
        """
        # 0. constants

        # 1. load encoder, model, optimizer
        encoder_device = HashEncoderNative(device=self.device).to(self.device)
        model_device = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_device.num_levels * 2).to(self.device)
        optimizer = torch.optim.RAdam([{'params': model_device.parameters(), 'weight_decay': 1e-6}, {'params': encoder_device.parameters(), 'eps': 1e-15}], lr=0.01, betas=(0.9, 0.99))

        # 2. load data to device
        train_video_data_device = load_videos_data_device(self.video_infos, dataset_type="train", device=self.device, dtype=self.dtype_device)  # (#videos, #frames, H, W, C)
        width, height = train_video_data_device.shape[3], train_video_data_device.shape[2]

        # 3. load poses
        train_indices = [0, 1, 2, 3]
        train_poses_device = torch.tensor([self.camera_infos[i].transform_matrices for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)

        # 4. pre-sample rays
        focals_device = torch.tensor([0.5 * width / torch.tan(0.5 * torch.tensor(self.camera_infos[i].camera_angle_x[0], dtype=self.dtype_device)) for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras)
        rays_origin_device, rays_direction_device, u_device, v_device = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=True)  # (#cameras, H, W, 3), (#cameras, H, W, 3), (H, W), (H, W)

        # 5. resample images
        train_video_resampled_device = resample_images_torch(train_video_data_device, u_device, v_device)  # (#videos, #frames, H, W, C)

        # 6. train


if __name__ == '__main__':
    hyfluid_video_infos.root_dir = "../data/hyfluid"
    hyfluid = HyFluidPipeline(hyfluid_video_infos, hyfluid_camera_infos_list, device=torch.device("cuda"), dtype_numpy=np.float32, dtype_device=torch.float16)
    hyfluid.train_density_device()
