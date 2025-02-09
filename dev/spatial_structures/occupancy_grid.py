import torch
import numpy as np


class OccupancyGrid():
    def __init__(self):
        self.grid = None
        self.resolution = None
        self.origin = None


def compute_frustum_and_check_points(camera_angle_x, transform, near, far, aspect_ratio, xyz_tensor):
    """
    计算视锥体并检查点是否在视锥体内
    :param camera_angle_x: 水平方向的 FOV（单位：弧度）
    :param transform: 4x4 相机变换矩阵（世界 -> 相机）
    :param near: 近平面距离
    :param far: 远平面距离
    :param aspect_ratio: 视口宽高比 w/h
    :param xyz_tensor: (N, 3) 形状的点云数据
    :return: (N,) bool 张量，表示每个点是否在视锥体内
    """
    # 计算焦距 f
    f = 0.5 / torch.tan(0.5 * camera_angle_x)  # 归一化焦距

    # 透视投影矩阵 (OpenGL 透视投影)
    P = torch.tensor([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=torch.float32)

    # 变换点到相机坐标系
    xyz_homo = torch.cat([xyz_tensor, torch.ones((xyz_tensor.shape[0], 1))], dim=1)  # (N, 4)
    xyz_camera = torch.matmul(transform, xyz_homo.T).T  # (N, 4)

    # 变换到 NDC 空间 (透视变换)
    xyz_clip = torch.matmul(P, xyz_camera.T).T  # (N, 4)
    xyz_ndc = xyz_clip[:, :3] / xyz_clip[:, 3:4]  # (N, 3)，进行透视除法

    # 判断点是否在 NDC [-1, 1] 范围内（即是否在视锥体内）
    in_frustum = (
            (xyz_ndc[:, 0] >= -1) & (xyz_ndc[:, 0] <= 1) &  # x 方向
            (xyz_ndc[:, 1] >= -1) & (xyz_ndc[:, 1] <= 1) &  # y 方向
            (xyz_ndc[:, 2] >= -1) & (xyz_ndc[:, 2] <= 1)  # z 方向（OpenGL 规范）
    )

    return in_frustum


# 示例：
camera_angle_x = torch.tensor(0.5)  # 假设 0.5 弧度
transform = torch.eye(4)  # 单位矩阵，表示没有变换
near, far = 0.1, 100.0
aspect_ratio = 16 / 9
xyz_tensor = torch.tensor([[0, 0, -1], [10, 10, -10], [-0.1, -0.1, -0.2]], dtype=torch.float32)

# 计算是否在视锥体内
inside_frustum = compute_frustum_and_check_points(camera_angle_x, transform, near, far, aspect_ratio, xyz_tensor)
print(inside_frustum)  # 输出哪些点在视锥体内
