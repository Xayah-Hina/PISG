import torch
import torchvision.io as io
import os
from pathlib import Path

from model.model_hyfluid import NeRFSmall, NeRFSmallPotential
from model.encoder_hyfluid import HashEncoderNativeFasterBackward

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


def load_cameras_data(*cameras_paths, ratio, device: torch.device, dtype: torch.dtype):
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

    focals = focals * ratio
    widths = widths * ratio
    heights = heights * ratio

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
    dirs_normalized = torch.nn.functional.normalize(dirs, dim=-1)

    return dirs_normalized, u, v


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


def sample_frustum(dirs: torch.Tensor, poses: torch.Tensor, batch_size: int, randomize: bool, device: torch.device):
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

    if randomize:
        indices = torch.randperm(num_rays, device=device)  # (N*H*W)
    else:
        indices = torch.arange(num_rays, device=device)  # (N*H*W)

    for i in range(0, num_rays, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_rays_o = rays_o[batch_indices]  # (batch_size, 3)
        batch_rays_d = rays_d[batch_indices]  # (batch_size, 3)
        yield batch_indices, batch_rays_o, batch_rays_d


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


def pos_world2smoke(inputs_pts, s_w2s, s_scale):
    pts_world_homo = torch.cat([inputs_pts, torch.ones_like(inputs_pts[..., :1])], dim=-1)
    pts_sim_ = torch.matmul(s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
    pts_sim = pts_sim_ / s_scale
    return pts_sim


def isInside(inputs_pts, s_w2s, s_scale, s_min, s_max):
    target_pts = pos_world2smoke(inputs_pts, s_w2s, s_scale)
    above = torch.logical_and(target_pts[..., 0] >= s_min[0], target_pts[..., 1] >= s_min[1])
    above = torch.logical_and(above, target_pts[..., 2] >= s_min[2])
    below = torch.logical_and(target_pts[..., 0] <= s_max[0], target_pts[..., 1] <= s_max[1])
    below = torch.logical_and(below, target_pts[..., 2] <= s_max[2])
    outputs = torch.logical_and(below, above)
    return outputs


def insideMask(inputs_pts, s_w2s, s_scale, s_min, s_max, to_float=False):
    mask = isInside(inputs_pts, s_w2s, s_scale, s_min, s_max)
    return mask.to(torch.float) if to_float else mask


import tqdm

if __name__ == '__main__':
    # =============================================================================
    # Load Dataset  (HyFluid Scene)
    target_device = torch.device("cuda:0")
    target_dtype = torch.float32
    ratio = 0.5
    batch_size = 256
    depth_size = 192
    global_step = 1
    lrate = 0.01
    lrate_decay = 1000
    videos_data = load_videos_data(*training_videos, ratio=ratio, dtype=target_dtype)
    poses, focals, width, height, near, far = load_cameras_data(*camera_calibrations, ratio=ratio, device=target_device, dtype=target_dtype)

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
    VOXEL_TRAN = torch.tensor([
        [1.0, 0.0, 7.5497901e-08, 8.1816666e-02],
        [0.0, 1.0, 0.0, -4.4627272e-02],
        [7.5497901e-08, 0.0, -1.0, -4.9089999e-03],
        [0.0, 0.0, 0.0, 1.0]
    ], device=target_device, dtype=target_dtype)

    VOXEL_SCALE = torch.tensor([0.4909, 0.73635, 0.4909], device=target_device, dtype=target_dtype)

    NEAR_float, FAR_float = float(near[0].item()), float(far[0].item())


    # TO DELETE
    # =============================================================================

    def phase_1(videos_data_resampled, batch_indices, batch_rays_o, batch_rays_d):
        batch_time, batch_target_pixels = sample_random_frame(videos_data=videos_data_resampled, batch_indices=batch_indices, device=target_device, dtype=target_dtype)

        t_vals = torch.linspace(0., 1., steps=depth_size, device=target_device, dtype=target_dtype)
        t_vals = t_vals.view(1, depth_size)
        z_vals = NEAR_float * (1. - t_vals) + FAR_float * t_vals
        z_vals = z_vals.expand(batch_size, depth_size)

        mid_vals = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper_vals = torch.cat([mid_vals, z_vals[..., -1:]], -1)
        lower_vals = torch.cat([z_vals[..., :1], mid_vals], -1)
        t_rand = torch.rand(z_vals.shape, device=target_device, dtype=target_dtype)
        z_vals = lower_vals + (upper_vals - lower_vals) * t_rand

        batch_dist_vals = z_vals[..., 1:] - z_vals[..., :-1]  # [batch_size, N_depths-1]
        batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * z_vals[..., :, None]  # [batch_size, N_depths, 3]
        batch_points_time = torch.cat([batch_points, batch_time.expand(batch_points[..., :1].shape)], dim=-1)
        batch_points_time_flat = batch_points_time.reshape(-1, 4)

        bbox_mask = insideMask(batch_points_time_flat[..., :3], s_w2s, s_scale, s_min, s_max, to_float=False)
        batch_points_time_flat_filtered = batch_points_time_flat[bbox_mask]
        batch_points_time_flat_filtered.requires_grad = True

        h = encoder_d(batch_points_time_flat_filtered)
        raw_d = model_d(h)
        raw_flat = torch.zeros([batch_size * depth_size, 1], device=target_device, dtype=torch.float32)
        raw_flat[bbox_mask] = raw_d
        raw = raw_flat.reshape(batch_size, depth_size, 1)

        dists_cat = torch.cat([batch_dist_vals, torch.tensor([1e10], device=target_device).expand(batch_dist_vals[..., :1].shape)], -1)  # [batch_size, N_depths]
        dists_final = dists_cat * torch.norm(batch_rays_d[..., None, :], dim=-1)  # [batch_size, N_depths]

        rgb_trained = torch.ones(3, device=target_device) * (0.6 + torch.tanh(model_d.rgb) * 0.4)
        raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)
        noise = 0.
        alpha = raw2alpha(raw[..., -1] + noise, dists_final)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=target_device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)
        img2mse = lambda x, y: torch.mean((x - y) ** 2)
        img_loss = img2mse(rgb_map, batch_target_pixels)

        return img_loss, h, batch_points_time_flat_filtered, raw_d


    def phase_2(batch_points_time_flat_filtered, _d_x, _d_y, _d_z, _d_t, raw_d):
        raw_vel, raw_f = model_v(encoder_v(batch_points_time_flat_filtered))
        _u_x, _u_y, _u_z, _u_t = None, None, None, None
        _u, _v, _w = raw_vel.split(1, dim=-1)
        split_nse = _d_t + (_u * _d_x + _v * _d_y + _w * _d_z)
        nse_errors = torch.mean(torch.square(split_nse))
        split_nse_wei = 0.001
        nseloss_fine = nse_errors * split_nse_wei

        proj_loss = torch.zeros_like(nseloss_fine)

        viz_dens_mask = raw_d.detach() > 0.1
        vel_norm = raw_vel.norm(dim=-1, keepdim=True)
        min_vel_mask = vel_norm.detach() < 0.2 * raw_d.detach()
        vel_reg_mask = min_vel_mask & viz_dens_mask
        min_vel_reg_map = (0.2 * raw_d - vel_norm) * vel_reg_mask.float()
        min_vel_reg = min_vel_reg_map.pow(2).mean()

        return nseloss_fine, proj_loss, min_vel_reg, nse_errors


    def g(x):
        return model_d(x)


    compile_phase_1 = torch.compile(phase_1, mode="max-autotune")
    compile_phase_2 = torch.compile(phase_2, mode="max-autotune")

    s_w2s = torch.inverse(VOXEL_TRAN).expand([4, 4])
    s_scale = VOXEL_SCALE.expand([3])
    s_min = torch.tensor([0.15, 0.0, 0.15], device=target_device, dtype=target_dtype)
    s_max = torch.tensor([0.85, 1.0, 0.85], device=target_device, dtype=target_dtype)

    for ITERATION in range(1, 2):
        dirs, u, v = shuffle_uv(focals=focals, width=int(width[0].item()), height=int(height[0].item()), randomize=True, device=torch.device("cpu"), dtype=target_dtype)
        videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(target_device)  # (T, V, H, W, C)
        dirs = dirs.to(target_device)

        for _2, (batch_indices, batch_rays_o, batch_rays_d) in enumerate(tqdm.tqdm(sample_frustum(dirs=dirs, poses=poses, batch_size=batch_size, randomize=True, device=target_device))):

            optimizer_d.zero_grad()
            optimizer_v.zero_grad()

            img_loss, h, batch_points_time_flat_filtered, raw_d = phase_1(videos_data_resampled, batch_indices, batch_rays_o, batch_rays_d)

            jac = torch.vmap(torch.func.jacrev(g))(h)
            jac_x = _get_minibatch_jacobian(h, batch_points_time_flat_filtered)
            jac = jac @ jac_x
            _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]

            nseloss_fine, proj_loss, min_vel_reg, nse_errors = phase_2(batch_points_time_flat_filtered, _d_x, _d_y, _d_z, _d_t, raw_d)

            if nse_errors.sum() > 10000:
                print(f'skip large loss {nse_errors.sum():.3g}, timestep={batch_points_time_flat_filtered[0, 3]}')
                continue

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
