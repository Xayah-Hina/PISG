import torch
import math


class OccupancyGrid():
    def __init__(self):
        self.grid = None
        self.resolution = None
        self.origin = None

def points_in_camera_frustum(
        points_world: torch.Tensor,
        cam_transform: torch.Tensor,
        camera_angle_x: float,
        width: int,
        height: int,
        near_clip: float,
        far_clip: float
):
    """
    判断一批三维点是否位于给定相机视锥体内，并返回世界坐标系下的视锥体8个角点。

    参数:
    -------
    points_world : (N, 3)或(N, 4) 的张量, 表示一批点的世界坐标.
    cam_transform: (4, 4) 的张量, 相机坐标 -> 世界坐标 的变换矩阵.
    camera_angle_x: float, 水平FOV(弧度).
    width, height : 图像宽度和高度.
    near_clip, far_clip: 近平面和远平面的距离.

    返回:
    -------
    visible_mask : (N,) 的布尔张量, True 表示该点在视锥体内, False 表示在视锥体外.
    frustum_corners_world : (8, 3) 的张量, 表示视锥体8个角点在世界坐标系下的位置(可用于可视化).
    """

    device = points_world.device
    dtype  = points_world.dtype

    # ---------------------------
    # 1) 计算垂直FOV
    # ---------------------------
    aspect = width / height  # 例: 1080/1920=0.5625
    camera_angle_y = 2.0 * math.atan(math.tan(camera_angle_x / 2.0) * (1.0 / aspect))

    # ---------------------------
    # 2) 计算相机坐标系下视锥体8个角点
    #    (假设相机面向 -Z)
    # ---------------------------
    # half angles
    half_angle_x = camera_angle_x * 0.5
    half_angle_y = camera_angle_y * 0.5

    tan_x = math.tan(half_angle_x)
    tan_y = math.tan(half_angle_y)

    # near plane corners (z = -near_clip)
    nx = near_clip * tan_x
    ny = near_clip * tan_y
    near_plane = torch.tensor([
        [ nx,  ny, -near_clip, 1.0],
        [ nx, -ny, -near_clip, 1.0],
        [-nx,  ny, -near_clip, 1.0],
        [-nx, -ny, -near_clip, 1.0],
    ], dtype=dtype, device=device)

    # far plane corners (z = -far_clip)
    fx = far_clip * tan_x
    fy = far_clip * tan_y
    far_plane = torch.tensor([
        [ fx,  fy, -far_clip, 1.0],
        [ fx, -fy, -far_clip, 1.0],
        [-fx,  fy, -far_clip, 1.0],
        [-fx, -fy, -far_clip, 1.0],
    ], dtype=dtype, device=device)

    # 合并近、远平面 -> (8,4)
    frustum_corners_cam = torch.cat([near_plane, far_plane], dim=0)

    # 将相机坐标系下的 corners 变换到世界坐标系
    # cam_transform : (cam->world)
    frustum_corners_world_h = frustum_corners_cam @ cam_transform.T  # (8,4)
    # 齐次除法
    frustum_corners_world = frustum_corners_world_h[:, :3] / frustum_corners_world_h[:, 3:4]

    # ---------------------------
    # 3) 判断点是否在视锥体内
    # ---------------------------
    # 如果输入是(N,3)，补齐为(N,4)
    if points_world.shape[-1] == 3:
        ones = torch.ones((points_world.shape[0], 1), device=device, dtype=dtype)
        points_world_h = torch.cat([points_world, ones], dim=-1)
    else:
        points_world_h = points_world

    # world -> camera 的逆矩阵
    M_world2cam = torch.inverse(cam_transform)  # (4,4)

    # 转到相机坐标系 (N,4)
    P_camera = points_world_h @ M_world2cam.T
    x_c = P_camera[:, 0]
    y_c = P_camera[:, 1]
    z_c = P_camera[:, 2]
    # w_c = P_camera[:, 3] # 如不是1, 需再除

    # -z 范围: [near_clip, far_clip]
    in_near = (-z_c >= near_clip)
    in_far  = (-z_c <= far_clip)

    # 水平范围: |x_c / -z_c| <= tan(half_angle_x)
    ratio_x = x_c / (-z_c + 1e-8)
    in_hfov = (torch.abs(ratio_x) <= tan_x)

    # 垂直范围: |y_c / -z_c| <= tan(half_angle_y)
    ratio_y = y_c / (-z_c + 1e-8)
    in_vfov = (torch.abs(ratio_y) <= tan_y)

    # 组合可见性
    visible_mask = in_near & in_far & in_hfov & in_vfov

    # 返回 (可见mask, 视锥体8点世界坐标)
    return visible_mask, frustum_corners_world


# -----------------------
# 简单演示
# -----------------------
if __name__ == "__main__":
    # 相机参数
    camera_angle_x = 0.40746459248665245  # 水平FOV(弧度)
    near_clip = 1.1
    far_clip  = 1.5
    width, height = 1080, 1920

    # 相机变换 (相机->世界)
    cam_transform_data = [
        [ 0.48627835512161255, -0.24310240149497986, -0.8393059968948364,  -0.7697111964225769 ],
        [-0.01889985240995884,  0.9573688507080078,  -0.2882491946220398,   0.013170702382922173],
        [ 0.8735995292663574,   0.15603208541870117,  0.4609531760215759,   0.3249526023864746   ],
        [ 0.0,                  0.0,                  0.0,                  1.0                  ]
    ]
    cam_transform = torch.tensor(cam_transform_data, dtype=torch.float32)

    # 准备一些测试点(世界坐标)
    # 这里随便造几组点, 有的在视锥体内, 有的在外
    points_world = torch.tensor([
        [ 0.0,   0.0,   0.0  ],  # 肯定离相机很近
        [0.32138651609420776, 0.3878946304321289, -0.27428650856018066],
        [ 0.0,   1.0,   0.0  ],
        [-0.7,   0.2,   2.0  ],
        [ 0.1,  -0.2,   2.5  ],
        [ 2.0,   2.0,   2.0  ],
    ], dtype=torch.float32)

    # 进行判断
    visible_mask, frustum_corners_world = points_in_camera_frustum(
        points_world,
        cam_transform,
        camera_angle_x,
        width,
        height,
        near_clip,
        far_clip
    )

    # 打印结果
    print("=== Points Visibility ===")
    for i, pt in enumerate(points_world):
        print(f"Point {pt.tolist()} -> in_frustum = {visible_mask[i].item()}")

    print("\n=== Frustum 8 Corners in World ===")
    print(frustum_corners_world)
