from dataloaders.dataloader_hyfluid import VideoInfos, CameraInfos, hyfluid_video_infos, hyfluid_camera_infos_list, load_videos_data_high_memory
from utils.utils_nerf import generate_rays_numpy, generate_rays_device, resample_images_scipy, resample_images_torch
import torch
import numpy as np
import time
import psutil

hyfluid_video_infos.root_dir = "../data/hyfluid"
dtype_numpy = np.float32
dtype_device = torch.float16


# 记录资源占用信息
def log_info(step_name):
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 ** 2)  # 转换为 MB
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0  # 转换为 MB
    print(f"[{step_name}] 时间: {time.perf_counter():.6f} 秒, CPU 内存: {cpu_memory:.2f} MB, GPU 显存: {gpu_memory:.2f} MB")


def profile_cpu():
    torch.cuda.empty_cache()  # 清理显存
    start_time = time.perf_counter()

    # 1. 加载数据
    step_start = time.perf_counter()
    train_video_data_numpy = load_videos_data_high_memory(hyfluid_video_infos, dataset_type="train").astype(dtype_numpy) / 255.0  # (#videos, #frames, H, W, C)
    log_info("加载数据")
    print(f"  ->  耗时: {time.perf_counter() - step_start:.6f} 秒")

    width, height = train_video_data_numpy.shape[3], train_video_data_numpy.shape[2]

    # 2. 计算相机参数
    step_start = time.perf_counter()
    train_indices = [0, 1, 2, 3]
    train_poses_numpy = np.array([hyfluid_camera_infos_list[i].transform_matrices for i in train_indices])  # (#cameras, 4, 4)
    focals_numpy = np.array([0.5 * width / np.tan(0.5 * hyfluid_camera_infos_list[i].camera_angle_x[0]) for i in train_indices], dtype=dtype_numpy)  # (#cameras,)
    log_info("计算相机参数")
    print(f"  ->  耗时: {time.perf_counter() - step_start:.6f} 秒")

    # 3. 生成光线
    step_start = time.perf_counter()
    rays_origin_numpy, rays_direction_numpy, u_numpy, v_numpy = generate_rays_numpy(train_poses_numpy, focals_numpy, width, height, randomize=True)
    log_info("生成光线 (generate_rays_numpy)")
    print(f"  ->  耗时: {time.perf_counter() - step_start:.6f} 秒")

    # 4. 重新采样视频
    step_start = time.perf_counter()
    train_video_resampled_numpy = resample_images_scipy(train_video_data_numpy, u_numpy, v_numpy)  # (#videos, #frames, H, W, C)
    log_info("重新采样视频 (resample_images_scipy)")
    print(f"  ->  耗时: {time.perf_counter() - step_start:.6f} 秒")

    total_time = time.perf_counter() - start_time
    print(f"=== 总运行时间: {total_time:.6f} 秒 ===")


if __name__ == "__main__":
    profile_cpu()
