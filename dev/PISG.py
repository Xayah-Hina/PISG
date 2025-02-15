from dataloaders.dataloader_hyfluid import (
    VideoInfos,
    load_videos_data_high_memory_numpy,
)

from pathlib import Path
import torch
import numpy as np
import tqdm
import os


class PISGPipeline:
    def __init__(self, video_infos: VideoInfos, device, dtype_numpy, dtype_device):
        self.video_infos = video_infos
        self.device = device
        self.dtype_numpy = dtype_numpy
        self.dtype_device = dtype_device

    def train_device(self):
        train_video_data_device = load_videos_data_high_memory_numpy(self.video_infos, dataset_type="train", dtype=self.dtype_numpy)
        print(train_video_data_device.shape)


if __name__ == '__main__':
    device = torch.device("cuda")
    infos = VideoInfos(
        root_dir=Path("../data/PISG/scene1"),
        train_videos=[Path("back.mp4"), Path("front.mp4"), Path("right.mp4"), Path("top.mp4")],
        validation_videos=[],
        test_videos=[],
    )
    camera_infos_path = [
        Path("cam_back.npz"),
        Path("cam_front.npz"),
        Path("cam_right.npz"),
        Path("cam_top.npz"),
    ]
    camera_infos = [np.load(infos.root_dir / path) for path in camera_infos_path]
    print(camera_infos[0]["cam_transform"])
    # PISG = PISGPipeline(infos, device=device, dtype_numpy=np.float32, dtype_device=torch.float32)
    # PISG.train_device()
