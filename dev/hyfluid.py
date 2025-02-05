from dataloaders.dataloader_hyfluid import VideoInfos, CameraInfos, hyfluid_video_infos, hyfluid_camera_infos_list, load_videos_data_device
from utils.utils_nerf import generate_rays_device, resample_images_torch
# from model.model_hyfluid import NeRFSmall, NeRFSmallPotential
# from model.encoder_hyfluid import HashEncoderHyFluid
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
        # 0. constants

        # 1. load data to device
        train_video_data_device = load_videos_data_device(self.video_infos, dataset_type="train", device=self.device, dtype=self.dtype_device)  # (#videos, #frames, H, W, C)
        width, height = train_video_data_device.shape[3], train_video_data_device.shape[2]

        # 2. load poses
        train_indices = [0, 1, 2, 3]
        train_poses_device = torch.tensor([self.camera_infos[i].transform_matrices for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)

        # 3. pre-sample rays
        focals_device = torch.tensor([0.5 * width / torch.tan(0.5 * torch.tensor(hyfluid_camera_infos_list[i].camera_angle_x[0], dtype=self.dtype_device)) for i in train_indices], device=self.device, dtype=self.dtype_device)  # (#cameras)
        rays_origin_device, rays_direction_device, u_device, v_device = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=True)

        # 4. resample images
        train_video_resampled_device = resample_images_torch(train_video_data_device, u_device, v_device)  # (#videos, #frames, H, W, C)

    # def train_density_only(self):
    #     # 0. constants
    #
    #     # 1. load data, move data to device
    #     train_video_data_device = torch.tensor(train_video_data_numpy, device=self.device, dtype=self.dtype_device)  # (#videos, #frames, H, W, C)
    #     width, height = train_video_data_numpy.shape[3], train_video_data_numpy.shape[2]
    #
    #     # 2. load poses
    #     train_indices = [0, 1, 2, 3]
    #     train_poses_numpy = np.array([self.camera_infos[i].transform_matrices for i in train_indices])  # (#cameras, 4, 4)
    #     train_poses_device = torch.tensor(train_poses_numpy, device=self.device, dtype=self.dtype_device)  # (#cameras, 4, 4)
    #
    #     # 3. pre-sample rays
    #     focals_numpy = np.array([0.5 * width / np.tan(0.5 * hyfluid_camera_infos_list[i].camera_angle_x[0]) for i in train_indices], dtype=self.dtype_numpy)  # (#cameras,)
    #     focals_device = torch.tensor(focals_numpy, device=self.device, dtype=self.dtype_device)  # (#cameras,)
    #     rays_origin_numpy, rays_direction_numpy, u_numpy, v_numpy = generate_rays_numpy(train_poses_numpy, focals_numpy, width, height, randomize=True)
    #     rays_origin_device, rays_direction_device, u_device, v_device = generate_rays_device(train_poses_device, focals=focals_device, width=width, height=height, randomize=True)
    #
    #     train_video_resampled_numpy = resample_images_scipy(train_video_data_numpy, u_numpy, v_numpy)  # (#videos, #frames, H, W, C)
    #     train_video_resampled_device = resample_images_torch(train_video_data_device, u_device, v_device)  # (#videos, #frames, H, W, C)
    #     print(train_video_resampled_numpy.shape, train_video_resampled_device.shape)
    #     print(np.allclose(train_video_resampled_numpy, train_video_resampled_device.cpu().numpy(), atol=1e-2, rtol=1e-2))

    def train_joint_velocity(self):
        pass

    def train_joint_vorticity(self):
        pass


if __name__ == '__main__':
    hyfluid_video_infos.root_dir = "../data/hyfluid"
    hyfluid = HyFluidPipeline(hyfluid_video_infos, hyfluid_camera_infos_list, device=torch.device("cuda"), dtype_numpy=np.float32, dtype_device=torch.float16)
    hyfluid.train_density_device()
