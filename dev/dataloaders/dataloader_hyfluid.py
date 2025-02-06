import torch
import torchvision.io as io
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from memory_profiler import memory_usage


# ====================================================================================================
# MAIN FUNCTION STARTS HERE ==========================================================================

@dataclass
class VideoInfos:
    root_dir: Path
    train_videos: list[Path]
    validation_videos: list[Path]
    test_videos: list[Path]


hyfluid_video_infos = VideoInfos(
    root_dir=Path("../../data/hyfluid"),
    train_videos=[Path("train00.mp4"), Path("train01.mp4"), Path("train02.mp4"), Path("train03.mp4")],
    validation_videos=[],
    test_videos=[Path("train04.mp4")],
)


@dataclass
class CameraInfos:
    transform_matrices: list[list[float]]
    camera_angle_x: list[float]
    near: list[float]
    far: list[float]


hyfluid_camera_infos_list = [
    CameraInfos(
        transform_matrices=[[0.48627835512161255, -0.24310240149497986, -0.8393059968948364, -0.7697111964225769],
                            [-0.01889985240995884, 0.9573688507080078, -0.2882491946220398, 0.013170702382922173],
                            [0.8735995292663574, 0.15603208541870117, 0.4609531760215759, 0.3249526023864746],
                            [0.0, 0.0, 0.0, 1.0]],
        camera_angle_x=[0.40746459248665245],
        near=[1.1],
        far=[1.5],
    ),  # train00.mp4
    CameraInfos(
        transform_matrices=[[0.8157652020454407, -0.1372431218624115, -0.5618642568588257, -0.39192497730255127],
                            [-0.04113851860165596, 0.9552109837532043, -0.2930521070957184, 0.010452679358422756],
                            [0.5769183039665222, 0.262175977230072, 0.7735819220542908, 0.8086869120597839],
                            [0.0, 0.0, 0.0, 1.0]],
        camera_angle_x=[0.39413608028840563],
        near=[1.1],
        far=[1.5],
    ),  # train01.mp4
    CameraInfos(
        transform_matrices=[[0.999511182308197, -0.0030406631994992495, -0.03111351653933525, 0.2844361364841461],
                            [-0.005995774641633034, 0.9581364989280701, -0.2862490713596344, 0.011681094765663147],
                            [0.03068138100206852, 0.28629571199417114, 0.9576499462127686, 0.9857829809188843],
                            [0.0, 0.0, 0.0, 1.0]],
        camera_angle_x=[0.41505697544547304],
        near=[1.1],
        far=[1.5],
    ),  # train02.mp4
    CameraInfos(
        transform_matrices=[[0.8836436867713928, 0.15215487778186798, 0.44274458289146423, 0.8974969983100891],
                            [-0.021659603342413902, 0.9579861760139465, -0.28599533438682556, 0.02680988796055317],
                            [-0.46765878796577454, 0.24312829971313477, 0.8498140573501587, 0.8316138386726379],
                            [0.0, 0.0, 0.0, 1.0]],
        camera_angle_x=[0.41320072172607875],
        near=[1.1],
        far=[1.5],
    ),  # train03.mp4
    CameraInfos(
        transform_matrices=[[0.6336104273796082, 0.20118704438209534, 0.7470352053642273, 1.2956339120864868],
                            [0.014488859102129936, 0.9623404741287231, -0.27146074175834656, 0.02436656318604946],
                            [-0.7735165357589722, 0.1828240603208542, 0.6068339943885803, 0.497546911239624],
                            [0.0, 0.0, 0.0, 1.0]],
        camera_angle_x=[0.40746459248665245],
        near=[1.1],
        far=[1.5],
    ),  # train04.mp4
]


def load_videos_data_high_memory_numpy(infos: VideoInfos, dataset_type: Literal["train", "validation", "test"], dtype) -> np.ndarray:
    """
    load videos data, for small dataset. (high memory consumption, but faster)

    Args:
    - infos: VideoInfos

    Returns:
    - np.ndarray of shape (#videos, #frames, H, W, C)

    """
    import imageio.v3 as imageio

    video_paths = {
        "train": infos.train_videos,
        "validation": infos.validation_videos,
        "test": infos.test_videos
    }.get(dataset_type)

    ##### validation layer start #####
    if video_paths is None: raise ValueError(f"Invalid dataset_type: {dataset_type}, expected one of ['train', 'validation', 'test']")
    if not video_paths: raise ValueError(f"No video paths found for dataset_type: {dataset_type}")
    ##### validation layer end #####

    _frames_arrays = []
    for video_path in video_paths:
        _path = infos.root_dir / video_path
        try:
            _frames = imageio.imread(_path, plugin="pyav")
            _frames_arrays.append(_frames)
        except Exception as e:
            print(f"Error loading video: {e}")

    return np.array(_frames_arrays).astype(dtype) / 255.0


def load_videos_data_low_memory_numpy(infos: VideoInfos, dataset_type: Literal["train", "validation", "test"], dtype) -> np.ndarray:
    """
    load videos data, for large dataset. (low memory consumption, but much slower)

    Args:
    - infos: VideoInfos

    Returns:
    - np.ndarray of shape (#videos, #frames, H, W, C)

    """
    import imageio.v2 as imageio

    video_paths = {
        "train": infos.train_videos,
        "validation": infos.validation_videos,
        "test": infos.test_videos
    }.get(dataset_type)

    ##### validation layer start #####
    if video_paths is None: raise ValueError(f"Invalid dataset_type: {dataset_type}, expected one of ['train', 'validation', 'test']")
    if not video_paths: raise ValueError(f"No video paths found for dataset_type: {dataset_type}")
    ##### validation layer end #####

    _frames_arrays = []
    for video_path in video_paths:
        _path = infos.root_dir / video_path
        try:
            _reader = imageio.get_reader(_path)
            _frames = [frame for frame in _reader]
            _frames_arrays.append(_frames)
        except Exception as e:
            print(f"Error loading video: {e}")

    return np.array(_frames_arrays).astype(dtype) / 255.0


def load_videos_data_device(infos: VideoInfos, dataset_type: Literal["train", "validation", "test"], device, dtype) -> torch.Tensor:
    """
    load videos data purely on device

    Args:
    - infos: VideoInfos
    - dataset_type: Literal["train", "validation", "test"]
    - device: torch.device
    - dtype: torch.dtype

    Returns:
    - torch.Tensor of shape (#videos, #frames, H, W, C)

    """
    video_paths = {
        "train": infos.train_videos,
        "validation": infos.validation_videos,
        "test": infos.test_videos
    }.get(dataset_type)

    ##### validation layer start #####
    if video_paths is None:
        raise ValueError(f"Invalid dataset_type: {dataset_type}, expected one of ['train', 'validation', 'test']")
    if not video_paths:
        raise ValueError(f"No video paths found for dataset_type: {dataset_type}")
    for video_path in video_paths:
        _path = os.path.normpath(infos.root_dir / video_path)
        if not Path(_path).exists():
            raise FileNotFoundError(f"Video file not found: {_path}")
    ##### validation layer end #####

    _frames_tensors = []
    for video_path in video_paths:
        _path = os.path.normpath(infos.root_dir / video_path)
        try:
            _frames, _, _ = io.read_video(_path, pts_unit="sec")
            _frames = _frames.to(device, dtype=dtype) / 255.0  # normalize to [0, 1]
            _frames_tensors.append(_frames)
        except Exception as e:
            print(f"Error loading video: {e}")

    return torch.stack(_frames_tensors)  # (V, T, H, W, C)


def resample_images_by_ratio_numpy(images: np.ndarray, ratio: float) -> np.ndarray:
    """
    Resample images by ratio using NumPy and OpenCV.

    Args:
    - images: np.ndarray of shape (V, T, H, W, C)
    - ratio: float, resampling ratio

    Returns:
    - np.ndarray of shape (V, T, H * ratio, W * ratio, C)
    """
    import cv2
    V, T, H, W, C = images.shape
    H_new, W_new = int(H * ratio), int(W * ratio)

    resampled_images = np.empty((V, T, H_new, W_new, C), dtype=images.dtype)

    for v in range(V):
        for t in range(T):
            resampled_images[v, t] = cv2.resize(images[v, t], (W_new, H_new), interpolation=cv2.INTER_LINEAR)

    return resampled_images


def resample_images_by_ratio_device(images: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    resample images by ratio

    Args:
    - images: torch.Tensor of shape (V, T, H, W, C)
    - ratio: float, resampling ratio

    Returns:
    - torch.Tensor of shape (V, T, H * ratio, W * ratio, C)

    """
    V, T, H, W, C = images.shape
    images_permuted = images.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W)  # (V * T, C, H, W)
    images_permuted_resized = (torch.nn.functional.interpolate(images_permuted, size=(int(H * ratio), int(W * ratio)), mode='bilinear', align_corners=False)
                               .reshape(V, T, C, int(H * ratio), int(W * ratio))
                               .permute(0, 1, 3, 4, 2))  # (V, T, H * ratio, W * ratio, C)
    return images_permuted_resized


# MAIN FUNCTION ENDS HERE ============================================================================
# ====================================================================================================


# ====================================================================================================
# UNIT TESTS START HERE ==============================================================================


def profile_memories():
    print("========== Memory Unit tests start ==========")

    video_infos = hyfluid_video_infos
    print(f"load_videos_data_high_memory, train: {memory_usage((load_videos_data_high_memory_numpy, (video_infos, 'train', np.float32)))}")
    print(f"load_videos_data_high_memory, test: {memory_usage((load_videos_data_high_memory_numpy, (video_infos, 'test', np.float32)))}")
    print(f"load_videos_data_low_memory, train: {memory_usage((load_videos_data_low_memory_numpy, (video_infos, 'train', np.float32)))}")
    print(f"load_videos_data_low_memory, test: {memory_usage((load_videos_data_low_memory_numpy, (video_infos, 'test', np.float32)))}")

    torch.cuda.empty_cache()

    initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 初始 GPU 内存 (MB)
    ret = load_videos_data_device(video_infos, "train", device=torch.device("cuda"), dtype=torch.float16)
    final_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 运行后 GPU 内存 (MB)
    print(f"load_videos_data_device 增加的 GPU 显存: {final_gpu_memory - initial_gpu_memory:.2f} MB")

    print("========== Memory Unit tests passed ==========")


def profile_times():
    print("========== Time Unit tests start ==========")
    import time

    video_infos = hyfluid_video_infos

    def time_function(name, func, *args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        t = end - start
        print(f"{name}: {t:.4f} seconds")

    time_function("high_mem_train", load_videos_data_high_memory_numpy, video_infos, "train", np.float32)
    time_function("high_mem_test", load_videos_data_high_memory_numpy, video_infos, "test", np.float32)
    time_function("low_mem_train", load_videos_data_low_memory_numpy, video_infos, "train", np.float32)
    time_function("low_mem_test", load_videos_data_low_memory_numpy, video_infos, "test", np.float32)
    time_function("device_train", load_videos_data_device, video_infos, "train", device=torch.device("cuda"), dtype=torch.float16)
    time_function("device_test", load_videos_data_device, video_infos, "test", device=torch.device("cuda"), dtype=torch.float16)

    print("========== Time Unit tests passed ==========")


if __name__ == "__main__":
    profile_memories()
    profile_times()

# UNIT TESTS END HERE ================================================================================
# ====================================================================================================
