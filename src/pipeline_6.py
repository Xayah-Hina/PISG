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


def shuffle_uv(focals: torch.Tensor, width: int, height: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    """
    Shuffle the UV coordinates of the image plane.

    Args:
    - focals: torch.Tensor of shape (N)
    - width: int
    - height: int
    - randomize: bool

    Returns:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - u: torch.Tensor of shape (H, W)
    - v: torch.Tensor of shape (H, W)
    """
    focals = focals.to(device)
    u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=device, dtype=dtype), torch.linspace(0, height - 1, height, device=device, dtype=dtype), indexing='xy')  # (H, W), (H, W)
    if randomize:
        du, dv = torch.rand_like(u), torch.rand_like(v)  # (H, W), (H, W)
        u, v = torch.clip(u + du, 0, width - 1), torch.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
    u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)
    dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)

    return dirs, u, v


def resample_frames(frames: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """
    Resample frames using the given UV coordinates.

    Args:
    - frames: torch.Tensor of shape (..., H, W, C)
    - u: torch.Tensor of shape (N, H, W)
    - v: torch.Tensor of shape (N, H, W)

    Returns:
    - resampled_images: torch.Tensor of shape (N, T, H, W, C)
    """

    H, W, C = frames.shape[-3:]
    u_norm, v_norm = 2.0 * (u / (W - 1)) - 1, 2.0 * (v / (H - 1)) - 1
    grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)  # (1, H, W, 2)
    orig_shape = frames.shape
    reshaped_images = frames.reshape(-1, H, W, C).permute(0, 3, 1, 2)  # (batch, C, H, W)
    resampled = torch.nn.functional.grid_sample(reshaped_images, grid.expand(reshaped_images.shape[0], -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)
    resampled_images = resampled.permute(0, 2, 3, 1).reshape(orig_shape)
    return resampled_images


def sample_frustum(dirs: torch.Tensor, poses: torch.Tensor, near: float, far: float, depth: int, batch_size: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    """
    Sample points in the frustum of each camera.

    Args:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - poses: torch.Tensor of shape (N, 4, 4)
    - near: float
    - far: float
    - depth: int
    - batch_size: int
    - randomize: bool

    Yields:
    - points: torch.Tensor of shape (batch_size, depth, 3)
    """

    rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # (N, H, W, 3)

    rays_d = rays_d.reshape(-1, 3)  # (N*H*W, 3)
    rays_o = rays_o.reshape(-1, 3)  # (N*H*W, 3)
    num_rays = rays_d.shape[0]

    depths = torch.linspace(near, far, steps=depth, device=device, dtype=dtype).unsqueeze(0)  # (1, depth)

    if randomize:
        indices = torch.randperm(num_rays, device=device)  # (N*H*W)
    else:
        indices = torch.arange(num_rays, device=device)  # (N*H*W)

    for i in range(0, num_rays, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_rays_o = rays_o[batch_indices]  # (batch_size, 3)
        batch_rays_d = rays_d[batch_indices]  # (batch_size, 3)

        batch_depths = depths.clone()

        if randomize:
            midpoints = (depths[:, :-1] + depths[:, 1:]) / 2.0
            noise = (torch.rand_like(midpoints) - 0.5) * (far - near) / depth
            batch_depths[:, :-1] = midpoints + noise

        batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * batch_depths[:, :, None]  # (batch_size, depth, 3)
        yield batch_points, batch_depths, batch_indices, batch_rays_o, batch_rays_d


def sample_random_frame(videos_data: torch.Tensor, batch_indices: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """
    Sample a random frame from the given videos data.

    Args:
    - videos_data: torch.Tensor of shape (T, V, H, W, C)
    - batch_indices: torch.Tensor of shape (batch_size)

    Returns:
    - batch_time: torch.Tensor of shape (1)
    - batch_target_pixels: torch.Tensor of shape (batch_size, C)
    """
    frame = torch.rand((), device=device, dtype=dtype) * (videos_data.shape[0] - 1)
    frame_floor = torch.floor(frame).long()
    frame_ceil = frame_floor + 1
    frames_alpha = frame - frame_floor.to(frame.dtype)
    target_frame = (1 - frames_alpha) * videos_data[frame_floor] + frames_alpha * videos_data[frame_ceil]  # (V * H * W, C)
    target_frame = target_frame.reshape(-1, 3)
    batch_target_pixels = target_frame[batch_indices]  # (batch_size, C)
    batch_time = frame / (videos_data.shape[0] - 1)

    return batch_time, batch_target_pixels


# =============================================================================
# TO DELETE
import numpy as np

def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3, :3]), -1)  # 4.world to 3.target
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    pos_scale = new_pose / (scale_vector)  # 3.target to 2.simulation
    return pos_scale


class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=[0.15, 0.0, 0.15], in_max=[0.85, 1., 0.85], device=torch.device("cuda")):
        self.s_w2s = torch.tensor(smoke_tran_inv, device=device, dtype=torch.float32).expand([4, 4])
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = torch.tensor(smoke_scale.copy(), device=device, dtype=torch.float32).expand([3])
        self.s_min = torch.tensor(in_min, device=device, dtype=torch.float32)
        self.s_max = torch.tensor(in_max, device=device, dtype=torch.float32)

    def world2sim(self, pts_world):
        pts_world_homo = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
        pts_sim_ = torch.matmul(self.s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def world2sim_rot(self, pts_world):
        pts_sim_ = torch.matmul(self.s_w2s[:3, :3], pts_world[..., None]).squeeze(-1)
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def sim2world(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_sim_homo = torch.cat([pts_sim_, torch.ones_like(pts_sim_[..., :1])], dim=-1)
        pts_world = torch.matmul(self.s2w, pts_sim_homo[..., None]).squeeze(-1)[..., :3]
        return pts_world

    def sim2world_rot(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_world = torch.matmul(self.s2w[:3, :3], pts_sim_[..., None]).squeeze(-1)
        return pts_world

    def isInside(self, inputs_pts):
        target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
        above = torch.logical_and(target_pts[..., 0] >= self.s_min[0], target_pts[..., 1] >= self.s_min[1])
        above = torch.logical_and(above, target_pts[..., 2] >= self.s_min[2])
        below = torch.logical_and(target_pts[..., 0] <= self.s_max[0], target_pts[..., 1] <= self.s_max[1])
        below = torch.logical_and(below, target_pts[..., 2] <= self.s_max[2])
        outputs = torch.logical_and(below, above)
        return outputs

    def insideMask(self, inputs_pts, to_float=True):
        return self.isInside(inputs_pts).to(torch.float) if to_float else self.isInside(inputs_pts)

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


def get_raw(POINTS_TIME: torch.Tensor, DISTs: torch.Tensor, RAYs_D_FLAT: torch.Tensor, BBOX_MODEL, MODEL, ENCODER, device):
    assert POINTS_TIME.dim() == 3 and POINTS_TIME.shape[-1] == 4
    assert POINTS_TIME.shape[0] == DISTs.shape[0] == RAYs_D_FLAT.shape[0]
    POINTS_TIME_FLAT = POINTS_TIME.view(-1, POINTS_TIME.shape[-1])  # [batch_size * N_depths, 4]
    out_dim = 1
    bbox_mask = BBOX_MODEL.insideMask(POINTS_TIME_FLAT[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True
        assert False
    POINTS_TIME_FLAT_FINAL = POINTS_TIME_FLAT[bbox_mask]

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

    RAW_FLAT = torch.zeros([POINTS_TIME_FLAT.shape[0], out_dim], device=POINTS_TIME_FLAT.device, dtype=torch.float32)
    RAW_FLAT[bbox_mask] = raw_d
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
    VOXEL_TRAN_np = np.array([[ 1.0000000e+00,0.0000000e+00,7.5497901e-08,8.1816666e-02], [ 0.0000000e+00,1.0000000e+00,0.0000000e+00,-4.4627272e-02], [ 7.5497901e-08,0.0000000e+00,-1.0000000e+00,-4.9089999e-03], [ 0.0000000e+00,0.0000000e+00,0.0000000e+00,1.0000000e+00]])
    VOXEL_SCALE_np = np.array([0.4909,0.73635,0.4909 ])
    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)

    # TO DELETE
    # =============================================================================

    batch_size = 256
    time_size = 1
    depth_size = 192
    global_step = 1
    lrate = 0.01
    lrate_decay = 100000
    for ITERATION in range(1, 2):
        dirs, u, v = shuffle_uv(focals=focals, width=int(width[0].item()), height=int(height[0].item()), randomize=True, device=torch.device("cpu"), dtype=target_dtype)
        videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(target_device)  # (T, V, H, W, C)
        dirs = dirs.to(target_device)

        for _2, (_x, _y, batch_indices, batch_rays_o, batch_rays_d) in enumerate(tqdm.tqdm(sample_frustum(dirs=dirs, poses=poses, near=near[0].item(), far=far[0].item(), depth=depth_size, batch_size=batch_size, randomize=True, device=target_device, dtype=target_dtype))):
            batch_time, batch_target_pixels = sample_random_frame(videos_data=videos_data_resampled, batch_indices=batch_indices, device=target_device, dtype=target_dtype)
            BATCH_RAYs_O_gpu = batch_rays_o
            BATCH_RAYs_D_gpu = batch_rays_d
            TARGET_S_gpu = batch_target_pixels

            POINTS_gpu, DISTs_gpu = get_points(BATCH_RAYs_O_gpu, BATCH_RAYs_D_gpu, NEAR_float, FAR_float, depth_size, randomize=True)
            POINTS_TIME_gpu = torch.cat([POINTS_gpu, batch_time.expand(POINTS_gpu[..., :1].shape)], dim=-1)

            optimizer_d.zero_grad()
            optimizer_v.zero_grad()

            ## START
            RGB_MAP, _d_x, _d_y, _d_z, _d_t, pts, raw_d = get_raw(POINTS_TIME_gpu, DISTs_gpu, BATCH_RAYs_D_gpu, BBOX_MODEL_gpu, model_d, encoder_d, target_device)

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

            print(f"nseloss_fine: {nseloss_fine}, img_loss: {10000 * img_loss}, proj_loss: {proj_loss}, min_vel_reg: {10.0 * min_vel_reg}, vel_loss: {vel_loss}")

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
