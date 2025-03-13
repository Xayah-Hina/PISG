from pipeline import *


class HoudiniExecutor:
    def __init__(self, batch_size, torch_device: torch.device, torch_dtype: torch.dtype):
        self.pipeline = PISGPipelineTorch(torch_device=torch_device, torch_dtype=torch_dtype)

        self.videos_data = self.pipeline.load_videos_data(*training_videos, ratio=self.pipeline.ratio)  # (T, V, H, W, C)
        self.masks = get_filter_mask(video_tensor=self.videos_data)
        self.poses, self.focals, self.width, self.height, self.near, self.far = self.pipeline.load_cameras_data(*camera_calibrations)
        self.focals = self.focals * self.pipeline.ratio
        self.width = self.width * self.pipeline.ratio
        self.height = self.height * self.pipeline.ratio

        dirs, u, v = shuffle_uv(focals=self.focals, width=int(self.width[0].item()), height=int(self.height[0].item()), randomize=True, device=torch.device("cpu"), dtype=self.pipeline.dtype)
        self.videos_data_resampled = resample_frames(frames=self.videos_data, u=u, v=v).to(self.pipeline.device)  # (T, V, H, W, C)
        self.dirs = dirs.to(self.pipeline.device)

        self.batch_size = batch_size
        self.generator = sample_frustum_with_mask(dirs=self.dirs, poses=self.poses, mask=self.masks, near=self.near[0].item(), far=self.far[0].item(), depth=self.pipeline.depth, batch_size=self.batch_size, randomize=True, device=self.pipeline.device, dtype=self.pipeline.dtype)

    def forward_1_iter(self):
        try:
            batch_points, batch_depths, batch_indices = next(self.generator)
        except StopIteration:
            dirs, u, v = shuffle_uv(focals=self.focals, width=int(self.width[0].item()), height=int(self.height[0].item()), randomize=True, device=torch.device("cpu"), dtype=self.pipeline.dtype)
            self.videos_data_resampled = resample_frames(frames=self.videos_data, u=u, v=v).to(self.pipeline.device)  # (T, V, H, W, C)
            self.dirs = dirs.to(self.pipeline.device)
            self.generator = sample_frustum_with_mask(dirs=self.dirs, poses=self.poses, mask=self.masks, near=self.near[0].item(), far=self.far[0].item(), depth=self.pipeline.depth, batch_size=self.batch_size, randomize=True, device=self.pipeline.device, dtype=self.pipeline.dtype)
            batch_points, batch_depths, batch_indices = next(self.generator)

        batch_time, batch_target_pixels = sample_random_frame(videos_data=self.videos_data_resampled, batch_indices=batch_indices, device=self.pipeline.device, dtype=self.pipeline.dtype)  # (batch_size, C)
        batch_rgb_map = self.pipeline.compiled_volume_render(batch_points, batch_depths, batch_time, self.poses, self.focals, self.width, self.height, self.near, self.far)
        loss_image = torch.nn.functional.mse_loss(batch_rgb_map, batch_target_pixels)
        self.pipeline.optimizer.zero_grad()
        loss_image.backward()
        self.pipeline.optimizer.step()
        self.pipeline.scheduler.step()

        return loss_image.item()
