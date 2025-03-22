from dataclasses import asdict

import torch
import torchvision.io as io
import torch.multiprocessing as mp
import os
import math
from pathlib import Path

from model.model_hyfluid import NeRFSmall, NeRFSmallPotential
from model.encoder_hyfluid import HashEncoderNativeFasterBackward

import tqdm


def find_relative_paths(relative_path_list):
    current_dir = Path.cwd()
    search_dirs = [current_dir, current_dir.parent, current_dir.parent.parent]

    for i in range(len(relative_path_list)):
        found = False
        relative_path = relative_path_list[i]
        for directory in search_dirs:
            full_path = directory / relative_path
            if full_path.exists():
                relative_path_list[i] = str(full_path.resolve())
                found = True
                break

        if not found:
            raise FileNotFoundError(f"file not found: {relative_path}")


# HyFluid Scene
training_videos_hyfluid = [
    "data/hyfluid/train00.mp4",
    "data/hyfluid/train01.mp4",
    "data/hyfluid/train02.mp4",
    "data/hyfluid/train03.mp4",
    "data/hyfluid/train04.mp4",
]
scene_min_hyfluid = [-0.132113, -0.103114, -0.753138]
scene_max_hyfluid = [0.773877, 0.99804, 0.186818]

camera_calibrations_hyfluid = [
    "data/hyfluid/cam_train00.npz",
    "data/hyfluid/cam_train01.npz",
    "data/hyfluid/cam_train02.npz",
    "data/hyfluid/cam_train03.npz",
    "data/hyfluid/cam_train04.npz",
]

training_videos = training_videos_hyfluid
camera_calibrations = camera_calibrations_hyfluid
scene_min_current = scene_min_hyfluid
scene_max_current = scene_max_hyfluid

find_relative_paths(training_videos)
find_relative_paths(camera_calibrations)


def load_videos_data(*video_paths, ratio: float, dtype: torch.dtype):
    """
    Load multiple videos directly from given paths onto the specified device, resample images by ratio.

    Args:
    - *paths: str (arbitrary number of video file paths)

    Returns:
    - torch.Tensor of shape (T, V, H * ratio, W * ratio, C)
    """

    if not video_paths:
        raise ValueError("No video paths provided.")

    valid_paths = []
    for video_path in video_paths:
        _path = os.path.normpath(video_path)
        if not Path(_path).exists():
            raise FileNotFoundError(f"Video path {_path} does not exist.")
        valid_paths.append(_path)

    _frames_tensors = []
    for _path in valid_paths:
        try:
            _frames, _, _ = io.read_video(_path, pts_unit="sec")
            _frames = _frames.to(dtype=dtype) / 255.0
            _frames_tensors.append(_frames)
        except Exception as e:
            print(f"Error loading video '{_path}': {e}")

    videos = torch.stack(_frames_tensors)

    V, T, H, W, C = videos.shape
    videos_permuted = videos.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W)
    new_H, new_W = int(H * ratio), int(W * ratio)
    videos_resampled = torch.nn.functional.interpolate(videos_permuted, size=(new_H, new_W), mode='bilinear', align_corners=False)
    videos_resampled = videos_resampled.reshape(V, T, C, new_H, new_W).permute(1, 0, 3, 4, 2)

    return videos_resampled


def load_cameras_data(*cameras_paths, device: torch.device, dtype: torch.dtype):
    """
    Load multiple camera calibration files directly from given paths onto the specified device.

    Args:
    - *cameras_paths: str (arbitrary number of camera calibration file paths)

    Returns:
    - poses: torch.Tensor of shape (N, 4, 4)
    - focals: torch.Tensor of shape (N)
    - width: torch.Tensor of shape (N)
    - height: torch.Tensor of shape (N)
    - near: torch.Tensor of shape (N)
    - far: torch.Tensor of shape (N)
    """

    if not cameras_paths:
        raise ValueError("No cameras paths provided.")

    valid_paths = []
    for camera_path in cameras_paths:
        _path = os.path.normpath(camera_path)
        if not Path(_path).exists():
            raise FileNotFoundError(f"Camera path {_path} does not exist.")
        valid_paths.append(_path)

    import numpy as np
    camera_infos = [np.load(path) for path in valid_paths]
    widths = [int(info["width"]) for info in camera_infos]
    assert len(set(widths)) == 1, f"Error: Inconsistent widths found: {widths}. All cameras must have the same resolution."
    heights = [int(info["height"]) for info in camera_infos]
    assert len(set(heights)) == 1, f"Error: Inconsistent heights found: {heights}. All cameras must have the same resolution."
    nears = [float(info["near"]) for info in camera_infos]
    assert len(set(nears)) == 1, f"Error: Inconsistent nears found: {nears}. All cameras must have the same near plane."
    fars = [float(info["far"]) for info in camera_infos]
    assert len(set(fars)) == 1, f"Error: Inconsistent fars found: {fars}. All cameras must have the same far plane."
    poses = torch.stack([torch.tensor(info["cam_transform"], device=device, dtype=dtype) for info in camera_infos])
    focals = torch.tensor([info["focal"] * widths[0] / info["aperture"] for info in camera_infos], device=device, dtype=dtype)
    widths = torch.tensor(widths, device=device, dtype=torch.int32)
    heights = torch.tensor(heights, device=device, dtype=torch.int32)
    nears = torch.tensor(nears, device=device, dtype=dtype)
    fars = torch.tensor(fars, device=device, dtype=dtype)

    return poses, focals, widths, heights, nears, fars


# =============================================================================
# TO DELETE
import numpy as np


def get_rays_np_continuous(H, W, c2w, K):
    # Generate random offsets for pixel coordinates
    random_offset = np.random.uniform(0, 1, size=(H, W, 2))
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    pixel_coords = np.stack((i, j), axis=-1) + random_offset

    # Clip pixel coordinates
    pixel_coords[..., 0] = np.clip(pixel_coords[..., 0], 0, W - 1)
    pixel_coords[..., 1] = np.clip(pixel_coords[..., 1], 0, H - 1)

    # Compute ray directions in camera space
    dirs = np.stack([
        (pixel_coords[..., 0] - K[0][2]) / K[0][0],
        -(pixel_coords[..., 1] - K[1][2]) / K[1][1],
        -np.ones_like(pixel_coords[..., 0])
    ], axis=-1)

    # Transform ray directions to world space
    rays_d = dirs @ c2w[:3, :3].T

    # Compute ray origins in world space
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)

    return rays_o, rays_d, pixel_coords[..., 0], pixel_coords[..., 1]


def sample_bilinear(img, xy):
    """
    Sample image with bilinear interpolation
    :param img: (T, V, H, W, 3)
    :param xy: (V, 2, H, W)
    :return: img: (T, V, H, W, 3)
    """
    T, V, H, W, _ = img.shape
    u, v = xy[:, 0], xy[:, 1]

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    u_floor, v_floor = np.floor(u).astype(int), np.floor(v).astype(int)
    u_ceil, v_ceil = np.ceil(u).astype(int), np.ceil(v).astype(int)

    u_ratio, v_ratio = u - u_floor, v - v_floor
    u_ratio, v_ratio = u_ratio[None, ..., None], v_ratio[None, ..., None]

    bottom_left = img[:, np.arange(V)[:, None, None], v_floor, u_floor]
    bottom_right = img[:, np.arange(V)[:, None, None], v_floor, u_ceil]
    top_left = img[:, np.arange(V)[:, None, None], v_ceil, u_floor]
    top_right = img[:, np.arange(V)[:, None, None], v_ceil, u_ceil]

    bottom = (1 - u_ratio) * bottom_left + u_ratio * bottom_right
    top = (1 - u_ratio) * top_left + u_ratio * top_right

    interpolated = (1 - v_ratio) * bottom + v_ratio * top

    return interpolated


def do_resample_rays(IMAGE_TRAIN_np, POSES_TRAIN_np, H, W, K, device):
    rays_list = []
    ij = []
    for p in POSES_TRAIN_np[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, p, K)
        rays_list.append([r_o, r_d])
        ij.append([i_, j_])
    ij = np.stack(ij, 0)
    images_train_sample = sample_bilinear(IMAGE_TRAIN_np, ij)
    ret_IMAGE_TRAIN_gpu = torch.tensor(images_train_sample, device=device, dtype=torch.float32).flatten(start_dim=1, end_dim=3)

    rays_np = np.stack(rays_list, 0)
    rays_np = np.transpose(rays_np, [0, 2, 3, 1, 4])
    rays_np = np.reshape(rays_np, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays_np = rays_np.astype(np.float32)
    ret_RAYs_gpu = torch.tensor(rays_np, device=device, dtype=torch.float32)
    ret_RAY_IDX_gpu = torch.randperm(ret_RAYs_gpu.shape[0], device=device, dtype=torch.int32)

    return ret_IMAGE_TRAIN_gpu, ret_RAYs_gpu, ret_RAY_IDX_gpu


def get_ray_batch(RAYs: torch.Tensor, RAYs_IDX: torch.Tensor, start: int, end: int):
    BATCH_RAYs_IDX = RAYs_IDX[start:end]  # [batch_size]
    BATCH_RAYs_O, BATCH_RAYs_D = torch.transpose(RAYs[BATCH_RAYs_IDX], 0, 1)  # [batch_size, 3]
    return BATCH_RAYs_O, BATCH_RAYs_D, BATCH_RAYs_IDX


def get_frames_at_times(IMAGEs: torch.Tensor, N_frames: int, N_times: int):
    assert N_frames > 1
    TIMEs_IDX = torch.randperm(N_frames, device=IMAGEs.device, dtype=torch.float32)[:N_times] + torch.randn(N_times, device=IMAGEs.device, dtype=torch.float32)  # [N_times]
    TIMEs_IDX_FLOOR = torch.clamp(torch.floor(TIMEs_IDX).long(), 0, N_frames - 1)  # [N_times]
    TIMEs_IDX_CEIL = torch.clamp(torch.ceil(TIMEs_IDX).long(), 0, N_frames - 1)  # [N_times]
    TIMEs_IDX_RESIDUAL = TIMEs_IDX - TIMEs_IDX_FLOOR.float()  # [N_times]
    TIME_STEPs = TIMEs_IDX / (N_frames - 1)  # [N_times]

    FRAMES_INTERPOLATED = IMAGEs[TIMEs_IDX_FLOOR] * (1 - TIMEs_IDX_RESIDUAL).view(-1, 1, 1) + IMAGEs[TIMEs_IDX_CEIL] * TIMEs_IDX_RESIDUAL.view(-1, 1, 1)
    return FRAMES_INTERPOLATED, TIME_STEPs


def get_points(RAYs_O: torch.Tensor, RAYs_D: torch.Tensor, near: float, far: float, N_depths: int, randomize: bool):
    T_VALs = torch.linspace(0., 1., steps=N_depths, device=RAYs_D.device, dtype=torch.float32)  # [N_depths]
    Z_VALs = near * torch.ones_like(RAYs_D[..., :1]) * (1. - T_VALs) + far * torch.ones_like(RAYs_D[..., :1]) * T_VALs  # [batch_size, N_depths]

    if randomize:
        MID_VALs = .5 * (Z_VALs[..., 1:] + Z_VALs[..., :-1])  # [batch_size, N_depths-1]
        UPPER_VALs = torch.cat([MID_VALs, Z_VALs[..., -1:]], -1)  # [batch_size, N_depths]
        LOWER_VALs = torch.cat([Z_VALs[..., :1], MID_VALs], -1)  # [batch_size, N_depths]
        T_RAND = torch.rand(Z_VALs.shape, device=RAYs_D.device, dtype=torch.float32)  # [batch_size, N_depths]
        Z_VALs = LOWER_VALs + (UPPER_VALs - LOWER_VALs) * T_RAND  # [batch_size, N_depths]

    DIST_VALs = Z_VALs[..., 1:] - Z_VALs[..., :-1]  # [batch_size, N_depths-1]
    POINTS = RAYs_O[..., None, :] + RAYs_D[..., None, :] * Z_VALs[..., :, None]  # [batch_size, N_depths, 3]
    return POINTS, DIST_VALs


def PDE_EQs(D_t, D_x, D_y, D_z, U, F, U_t=None, U_x=None, U_y=None, U_z=None, detach=False):
    eqs = []
    dts = [D_t]
    dxs = [D_x]
    dys = [D_y]
    dzs = [D_z]

    F = torch.cat([torch.zeros_like(F[:, :1]), F], dim=1) * 0  # (N,4)
    u, v, w = U.split(1, dim=-1)  # (N,1)
    F_t, F_x, F_y, F_z = F.split(1, dim=-1)  # (N,1)
    dfs = [F_t, F_x, F_y, F_z]

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim=-1)  # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim=-1)  # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim=-1)  # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim=-1)  # [d_z, u_z, v_z, w_z]
    else:
        dfs = [F_t]

    for i, (dt, dx, dy, dz, df) in enumerate(zip(dts, dxs, dys, dzs, dfs)):
        if i == 0:
            _e = dt + (u * dx + v * dy + w * dz) + df
        else:
            if detach:
                _e = dt + (u.detach() * dx + v.detach() * dy + w.detach() * dz) + df
            else:
                _e = dt + (u * dx + v * dy + w * dz) + df
        eqs += [_e]

    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [dxs[1] + dys[2] + dzs[3]]

    return eqs


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)

        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def get_ray_pts_velocity_and_derivitives(
        pts,
        network_vel_fn,
        N_samples,
        **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    if kwargs['no_vel_der']:
        vel_output, f_output = network_vel_fn(pts)
        ret = {}
        ret['raw_vel'] = vel_output
        ret['raw_f'] = f_output
        return ret

    def g(x):
        return model(x)[0]

    model = kwargs['network_fn']
    embed_fn = kwargs['embed_fn']
    h = embed_fn(pts)
    vel_output, f_output = model(h)
    ret = {}
    ret['raw_vel'] = vel_output
    ret['raw_f'] = f_output
    if not kwargs['no_vel_der']:
        jac = torch.vmap(torch.func.jacrev(g))(h)
        jac_x = _get_minibatch_jacobian(h, pts)
        jac = jac @ jac_x
        assert jac.shape == (pts.shape[0], 3, 4)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
        d = _u_x[:, 0] + _u_y[:, 1] + _u_z[:, 2]
        ret['raw_vel'] = vel_output
        ret['_u_x'] = _u_x
        ret['_u_y'] = _u_y
        ret['_u_z'] = _u_z
        ret['_u_t'] = _u_t

    return ret


def batchify_get_ray_pts_velocity_and_derivitive(pts, chunk=1024 * 64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, pts.shape[0], chunk):
        ret = get_ray_pts_velocity_and_derivitives(pts[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def get_velocity_and_derivitives(pts,
                                 **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: float. Focal length of pinhole camera.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    # Render and reshape
    all_ret = batchify_get_ray_pts_velocity_and_derivitive(pts, **kwargs)

    k_extract = ['raw_vel', 'raw_f'] if kwargs['no_vel_der'] else ['raw_vel', 'raw_f', '_u_x', '_u_y', '_u_z', '_u_t']
    ret_list = [all_ret[k] for k in k_extract]
    return ret_list


def get_raw(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor, MODEL, ENCODER, device):
    assert POINTS_TIME.dim() == 3 and POINTS_TIME.shape[-1] == 4
    assert POINTS_TIME.shape[0] == DISTs.shape[0] == RAYs_D_FLAT.shape[0]
    POINTS_TIME_FLAT = POINTS_TIME.view(-1, POINTS_TIME.shape[-1])  # [batch_size * N_depths, 4]
    out_dim = 1
    POINTS_TIME_FLAT_FINAL = POINTS_TIME_FLAT

    ## START
    POINTS_TIME_FLAT_FINAL.requires_grad = True

    def g(x):
        return MODEL(x)

    h = ENCODER(POINTS_TIME_FLAT_FINAL)
    raw_d = MODEL(h)
    jac = torch.vmap(torch.func.jacrev(g))(h)
    jac_x = _get_minibatch_jacobian(h, POINTS_TIME_FLAT_FINAL)
    jac = jac @ jac_x
    ret = {'raw_d': raw_d, 'pts': POINTS_TIME_FLAT_FINAL}
    _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]
    ret['_d_x'] = _d_x
    ret['_d_y'] = _d_y
    ret['_d_z'] = _d_z
    ret['_d_t'] = _d_t
    ## END

    RAW_FLAT = raw_d
    RAW = RAW_FLAT.reshape(*POINTS_TIME.shape[:-1], out_dim)
    assert RAW.dim() == 3 and RAW.shape[-1] == 1

    DISTs_cat = torch.cat([DISTs, torch.tensor([1e10], device=DISTs.device).expand(DISTs[..., :1].shape)], -1)  # [batch_size, N_depths]
    DISTS_final = DISTs_cat * torch.norm(RAYs_D_FLAT[..., None, :], dim=-1)  # [batch_size, N_depths]

    RGB_TRAINED = torch.ones(3, device=POINTS_TIME.device) * (0.6 + torch.tanh(MODEL.rgb) * 0.4)
    raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)
    noise = 0.
    alpha = raw2alpha(RAW[..., -1] + noise, DISTS_final)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * RGB_TRAINED, -2)
    return rgb_map, _d_x, _d_y, _d_z, _d_t, POINTS_TIME_FLAT_FINAL, raw_d


# TO DELETE
# =============================================================================


if __name__ == '__main__':
    # =============================================================================
    # Load Dataset  (HyFluid Scene)
    target_device = torch.device("cuda:0")
    target_dtype = torch.float32
    ratio = 0.5
    videos_data = load_videos_data(*training_videos, ratio=ratio, dtype=target_dtype)
    poses, focals, width, height, near, far = load_cameras_data(*camera_calibrations, device=target_device, dtype=target_dtype)
    focals = focals * ratio
    width = width * ratio
    height = height * ratio
    print(videos_data.shape)
    print(poses.shape)
    print(focals)
    print(width)
    print(height)
    print(near)
    print(far)
    # Load Dataset  (HyFluid Scene)
    # =============================================================================

    # =============================================================================
    # Model Initialization
    encoder_d = HashEncoderNativeFasterBackward().to(target_device)
    model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_d.num_levels * 2).to(target_device)
    optimizer_d = torch.optim.RAdam([{'params': model_d.parameters(), 'weight_decay': 1e-6}, {'params': encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
    encoder_v = HashEncoderNativeFasterBackward().to(target_device)
    model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_v.num_levels * 2, use_f=False).to(target_device)
    optimizer_v = torch.optim.RAdam([{'params': model_v.parameters(), 'weight_decay': 1e-6}, {'params': encoder_v.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
    # Model Initialization
    # =============================================================================

    # =============================================================================
    # TO DELETE
    NEAR_float, FAR_float = float(near[0].item()), float(far[0].item())
    H_int, W_int = int(height[0].item()), int(width[0].item())
    FOCAL_float = float(focals[0].item())
    K = np.array([[FOCAL_float, 0, 0.5 * W_int], [0, FOCAL_float, 0.5 * H_int], [0, 0, 1]])
    print(f"NEAR_float: {NEAR_float}, type: {type(NEAR_float)}, FAR_float: {FAR_float}, type: {type(FAR_float)}, H_int: {H_int}, type: {type(H_int)}, W_int: {W_int}, type: {type(W_int)}, FOCAL_float: {FOCAL_float}, type: {type(FOCAL_float)}")

    network_query_fn = lambda x: model_d(encoder_d(x))
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': 1.0,
        'N_samples': 192,
        'network_fn': model_d,
        'embed_fn': encoder_d,
        'near': NEAR_float,
        'far': FAR_float,
    }


    def network_vel_fn(x):
        with torch.enable_grad():
            v, f = model_v(encoder_v(x))
            return v, f


    render_kwargs_train_vel = {
        'network_vel_fn': network_vel_fn,
        'perturb': 1.0,
        'N_samples': 192,
        'network_fn': model_v,
        'embed_fn': encoder_v,
        'near': NEAR_float,
        'far': FAR_float,
    }
    # TO DELETE
    # =============================================================================

    batch_size = 256
    time_size = 1
    depth_size = 192
    global_step = 1
    lrate = 0.01
    lrate_decay = 100000
    for ITERATION in range(1, 2):
        IMAGE_TRAIN_gpu, RAYs_gpu, RAY_IDX_gpu = do_resample_rays(videos_data.cpu().numpy(), poses.cpu().numpy(), H_int, W_int, K, target_device)
        print(f"IMAGE_TRAIN_gpu shape: {IMAGE_TRAIN_gpu.shape}, RAYs_gpu shape: {RAYs_gpu.shape}, RAY_IDX_gpu shape: {RAY_IDX_gpu.shape}")

        for i in tqdm.trange(0, RAY_IDX_gpu.shape[0], batch_size):
            BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, BATCH_RAYs_IDX_gpu = get_ray_batch(RAYs_gpu, RAY_IDX_gpu, i, i + batch_size)  # [batch_size, 3], [batch_size, 3], [batch_size]
            FRAMES_INTERPOLATED_gpu, TIME_STEPs_gpu = get_frames_at_times(IMAGE_TRAIN_gpu, IMAGE_TRAIN_gpu.shape[0], time_size)  # [N_times, N x H x W, 3], [N_times]
            TARGET_S_gpu = FRAMES_INTERPOLATED_gpu[:, BATCH_RAYs_IDX_gpu].flatten(0, 1)  # [batch_size * N_times, 3]

            optimizer_d.zero_grad()
            optimizer_v.zero_grad()

            POINTS_gpu, DISTs_gpu = get_points(BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, NEAR_float, FAR_float, depth_size, randomize=True)  # [batch_size, N_depths, 3]

            for TIME_STEP_gpu in TIME_STEPs_gpu:
                POINTS_TIME_gpu = torch.cat([POINTS_gpu, TIME_STEP_gpu.expand(POINTS_gpu[..., :1].shape)], dim=-1)  # [batch_size, N_depths, 4]

                ## START
                RGB_MAP, _d_x, _d_y, _d_z, _d_t, pts, raw_d = get_raw(POINTS_TIME_gpu, DISTs_gpu, BATCH_RAYs_D_gpu, model_d, encoder_d, target_device)

                raw_vel, raw_f = get_velocity_and_derivitives(pts, no_vel_der=True, **render_kwargs_train_vel)
                _u_x, _u_y, _u_z, _u_t = None, None, None, None

                split_nse = PDE_EQs(_d_t, _d_x, _d_y, _d_z, raw_vel, raw_f, _u_t, _u_x, _u_y, _u_z, detach=False)
                nse_errors = [mean_squared_error(x, 0.0) for x in split_nse]
                if torch.stack(nse_errors).sum() > 10000:
                    print(f'skip large loss {torch.stack(nse_errors).sum():.3g}, timestep={pts[0, 3]}')
                    continue
                nseloss_fine = 0.0
                split_nse_wei = [0.001]

                img2mse = lambda x, y: torch.mean((x - y) ** 2)
                img_loss = img2mse(RGB_MAP, TARGET_S_gpu)
                for ei, wi in zip(nse_errors, split_nse_wei):
                    nseloss_fine = ei * wi + nseloss_fine

                proj_loss = torch.zeros_like(img_loss)

                viz_dens_mask = raw_d.detach() > 0.1
                vel_norm = raw_vel.norm(dim=-1, keepdim=True)
                min_vel_mask = vel_norm.detach() < 0.2 * raw_d.detach()
                vel_reg_mask = min_vel_mask & viz_dens_mask
                min_vel_reg_map = (0.2 * raw_d - vel_norm) * vel_reg_mask.float()
                min_vel_reg = min_vel_reg_map.pow(2).mean()
                # ipdb.set_trace()

                vel_loss = nseloss_fine + 10000 * img_loss + 1.0 * proj_loss + 10.0 * min_vel_reg
                vel_loss.backward()

                optimizer_d.step()
                optimizer_v.step()

                decay_rate = 0.1
                decay_steps = lrate_decay
                new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer_d.param_groups:
                    param_group['lr'] = new_lrate
                for param_group in optimizer_v.param_groups:
                    param_group['lr'] = new_lrate

                global_step += 1

        os.makedirs("checkpoint", exist_ok=True)
        path = os.path.join("checkpoint", 'den_{:06d}.tar'.format(ITERATION))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': model_d.state_dict(),
            'embed_fn_state_dict': encoder_d.state_dict(),
            'optimizer_state_dict': optimizer_d.state_dict(),
        }, path)
        path = os.path.join("checkpoint", 'vel_{:06d}.tar'.format(ITERATION))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': model_v.state_dict(),
            'embed_fn_state_dict': encoder_v.state_dict(),
            'optimizer_state_dict': optimizer_d.state_dict(),
        }, path)
