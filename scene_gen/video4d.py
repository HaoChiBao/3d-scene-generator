"""Monocular video → 4D Gaussians (canonical 3DGS + deformation).

This is the reconstruct half of CAT4D (Wu et al., CVPR 2024 4D-GS): posed
frames, then a time-conditioned Gaussian field. CAT4D's multi-view video
diffusion is not public, so we do not invent extra cameras.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - GPU image always has torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


ProgressFn = Callable[[str], None]


def positional_encoding(x, n_freqs: int):
    """NeRF-style PE: x plus sin/cos of 2^k * x. x is [..., D]."""
    if n_freqs <= 0:
        return x
    freqs = (2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)) * math.pi
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


def scale_intrinsics(
    K: np.ndarray,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> np.ndarray:
    """Scale a pinhole K from (H, W) src to dst."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = np.array(K, dtype=np.float32, copy=True)
    out[0, :] *= float(dst_w) / max(float(src_w), 1.0)
    out[1, :] *= float(dst_h) / max(float(src_h), 1.0)
    return out


def export_times(n: int) -> list[float]:
    """Evenly spaced t in [0, 1], inclusive."""
    if n <= 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


class DeformField(nn.Module if nn is not None else object):
    """MLP: PE(xyz) + PE(t) → Δxyz, Δquat, Δlog-scale.

    Last layer starts at 0 so t=any is identity before training.
    """

    def __init__(
        self,
        *,
        xyz_freqs: int = 6,
        time_freqs: int = 4,
        hidden: int = 128,
        depth: int = 4,
    ) -> None:
        super().__init__()
        in_dim = 3 + 6 * xyz_freqs + 1 + 2 * time_freqs
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(max(depth - 1, 0)):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU(inplace=True)])
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 3 + 4 + 3)
        self.xyz_freqs = xyz_freqs
        self.time_freqs = time_freqs
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, xyz: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if t.ndim == 0:
            t = t.expand(xyz.shape[0], 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] == 1 and xyz.shape[0] > 1:
            t = t.expand(xyz.shape[0], 1)
        enc = torch.cat(
            [
                positional_encoding(xyz, self.xyz_freqs),
                positional_encoding(t, self.time_freqs),
            ],
            dim=-1,
        )
        delta = self.out(self.net(enc))
        return delta[:, :3], delta[:, 3:7], delta[:, 7:10]


def reconstruct_video_4d(
    media_path: Path,
    output_dir: Path,
    *,
    max_frames: int = 24,
    target_fps: float = 4.0,
    resize_long_edge: int = 768,
    train_steps: int = 8000,
    n_times: int = 8,
    warmup_frac: float = 0.35,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Video → VGGT poses → 4D-GS snapshots."""
    if torch is None:
        raise RuntimeError("Video 4D needs PyTorch on the GPU worker")
    from scene_gen.frames import normalize_times, prepare_timed_folder

    def progress(msg: str) -> None:
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    progress("Extracting frames with timestamps…")
    image_paths, times_sec = prepare_timed_folder(
        media_path,
        work_dir,
        target_fps=float(target_fps),
        max_frames=int(max_frames),
    )
    if len(image_paths) < 3:
        raise ValueError(
            f"Need at least 3 frames for Video 4D, got {len(image_paths)}. "
            "Use a longer clip or raise sample fps."
        )
    image_paths = [_resize_long_edge(p, int(resize_long_edge)) for p in image_paths]
    times_norm = normalize_times(times_sec)
    duration_sec = float(max(times_sec) - min(times_sec))

    progress(f"Estimating cameras (VGGT) on {len(image_paths)} frames…")
    poses = _estimate_poses([str(p) for p in image_paths])
    if len(poses.point_cloud) < 100:
        raise ValueError(
            f"Only {len(poses.point_cloud)} points recovered. "
            "Need camera motion or a clearer clip."
        )

    progress(
        f"Training 4D Gaussians ({train_steps} steps: "
        f"static warmup, then deformation)…"
    )
    splats, deform, n_gaussians = _train_4d_gaussians(
        image_paths=[str(p) for p in image_paths],
        times_norm=times_norm,
        extrinsics=poses.extrinsics,
        intrinsics=poses.intrinsics,
        image_sizes=poses.image_sizes,
        point_cloud=poses.point_cloud,
        point_colors=poses.point_colors,
        output_dir=output_dir,
        train_steps=int(train_steps),
        warmup_frac=float(warmup_frac),
        on_progress=progress,
    )

    progress(f"Exporting {n_times} time snapshots…")
    timeline = _export_time_plys(
        splats,
        deform,
        output_dir,
        n_times=int(n_times),
        duration_sec=duration_sec,
    )

    elapsed = time.time() - t0
    meta = {
        "model": "video4d",
        "pipeline": "VGGT → 4D-GS (canonical + deform MLP)",
        "note": (
            "CAT4D reconstruct stage only. No multi-view video diffusion "
            "(weights are not public)."
        ),
        "has_gaussians": True,
        "n_gaussians": n_gaussians,
        "n_frames": len(image_paths),
        "n_times": len(timeline["times"]),
        "duration_sec": round(duration_sec, 3),
        "max_frames": max_frames,
        "target_fps": target_fps,
        "resize_long_edge": resize_long_edge,
        "train_steps": train_steps,
        "warmup_frac": warmup_frac,
        "elapsed_sec": round(elapsed, 1),
        "artifacts": {
            "gaussians": "gaussians.ply",
            "canonical": "canonical.ply",
            "timeline": "timeline.json",
            "times": [item["file"] for item in timeline["times"]],
        },
        "times": timeline["times"],
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (output_dir / "timeline.json").write_text(
        json.dumps(timeline, indent=2), encoding="utf-8"
    )
    progress(f"Done — {n_gaussians} Gaussians, {len(timeline['times'])} times, {elapsed:.0f}s")
    return meta


def _resize_long_edge(path: Path, long_edge: int) -> Path:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return path
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= long_edge or long_edge <= 0:
        return path
    scale = long_edge / float(current)
    resized = cv2.resize(
        image,
        (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    cv2.imwrite(str(path), resized)
    return path


def _estimate_poses(image_paths: list[str]):
    from recon3d.pose_estimation import estimate_poses_vggt

    n = len(image_paths)
    if n > 40:
        from recon3d.chunked_vggt import estimate_poses_chunked_vggt

        return estimate_poses_chunked_vggt(
            image_paths,
            device="cuda",
            conf_threshold=1.0,
            chunk_size=20,
            overlap=5,
            use_factor_graph=False,
        )
    return estimate_poses_vggt(
        image_paths,
        device="cuda",
        conf_threshold=1.0,
    )


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def _knn(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    from sklearn.neighbors import NearestNeighbors

    x_np = x.detach().cpu().numpy().astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
    distances, _ = nn.kneighbors(x_np)
    return torch.from_numpy(distances.astype(np.float32)).to(x.device)


def _init_splats(
    points: np.ndarray,
    colors: np.ndarray,
    sh_degree: int,
    device: torch.device,
    max_points: int,
) -> nn.ParameterDict:
    pts = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
    rgbs = torch.from_numpy(np.asarray(colors, dtype=np.float32)).to(device)
    if rgbs.max() > 1.5:
        rgbs = rgbs / 255.0
    if len(pts) > max_points:
        indices = torch.randperm(len(pts), device=device)[:max_points]
        pts = pts[indices]
        rgbs = rgbs[indices]
    dist2_avg = (_knn(pts, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg.clamp(min=1e-8))
    scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)
    n = pts.shape[0]
    quats = torch.rand((n, 4), device=device)
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    sh_coeffs = torch.zeros((n, (sh_degree + 1) ** 2, 3), device=device)
    sh_coeffs[:, 0, :] = _rgb_to_sh(rgbs)
    return nn.ParameterDict(
        {
            "means": nn.Parameter(pts),
            "scales": nn.Parameter(scales),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(opacities),
            "sh0": nn.Parameter(sh_coeffs[:, :1, :].contiguous()),
            "shN": nn.Parameter(sh_coeffs[:, 1:, :].contiguous()),
        }
    )


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean SSIM. pred/gt are [H, W, 3] in [0, 1]."""
    pred_n = pred.permute(2, 0, 1).unsqueeze(0)
    gt_n = gt.permute(2, 0, 1).unsqueeze(0)
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(pred_n, 11, 1, 5)
    mu_y = F.avg_pool2d(gt_n, 11, 1, 5)
    sig_x = F.avg_pool2d(pred_n * pred_n, 11, 1, 5) - mu_x * mu_x
    sig_y = F.avg_pool2d(gt_n * gt_n, 11, 1, 5) - mu_y * mu_y
    sig_xy = F.avg_pool2d(pred_n * gt_n, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2)
    )
    return ssim_map.mean()


def _apply_deform(
    splats: nn.ParameterDict,
    deform: DeformField,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_xyz, d_quat, d_scale = deform(splats["means"], t)
    means = splats["means"] + d_xyz
    quats = F.normalize(splats["quats"] + d_quat, dim=-1)
    scales = torch.exp(splats["scales"] + d_scale)
    return means, quats, scales, d_xyz


def _train_4d_gaussians(
    *,
    image_paths: list[str],
    times_norm: list[float],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_sizes: list[tuple[int, int]],
    point_cloud: np.ndarray,
    point_colors: np.ndarray,
    output_dir: Path,
    train_steps: int,
    warmup_frac: float,
    on_progress: ProgressFn,
    sh_degree: int = 3,
    init_points: int = 100_000,
    max_gaussians: int = 180_000,
) -> tuple[nn.ParameterDict, DeformField, int]:
    import cv2
    from gsplat import export_splats
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    device = torch.device("cuda")
    splats = _init_splats(point_cloud, point_colors, sh_degree, device, init_points)
    deform = DeformField().to(device)
    scene_scale = float(np.percentile(np.linalg.norm(point_cloud, axis=1), 95))
    if not math.isfinite(scene_scale) or scene_scale <= 1e-6:
        scene_scale = 1.0

    splat_optimizers = {
        "means": torch.optim.Adam(
            [splats["means"]], lr=1.6e-4 * scene_scale, eps=1e-15
        ),
        "scales": torch.optim.Adam([splats["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([splats["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2, eps=1e-15),
        "sh0": torch.optim.Adam([splats["sh0"]], lr=2.5e-3, eps=1e-15),
        "shN": torch.optim.Adam([splats["shN"]], lr=1.25e-4, eps=1e-15),
    }
    deform_opt = torch.optim.Adam(deform.parameters(), lr=8e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        splat_optimizers["means"], gamma=0.01 ** (1.0 / max(train_steps, 1))
    )
    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(splats, splat_optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    images_gt: list[torch.Tensor] = []
    viewmats: list[torch.Tensor] = []
    Ks: list[torch.Tensor] = []
    for i, path in enumerate(image_paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype(np.float32)
        images_gt.append(torch.from_numpy(img).to(device))
        w2c = extrinsics[i].astype(np.float32)
        if w2c.shape == (3, 4):
            full = np.eye(4, dtype=np.float32)
            full[:3, :] = w2c
            w2c = full
        viewmats.append(torch.from_numpy(w2c).to(device))
        src_hw = image_sizes[i] if i < len(image_sizes) else img.shape[:2]
        K = scale_intrinsics(intrinsics[i], src_hw, img.shape[:2])
        Ks.append(torch.from_numpy(K).to(device))

    times_t = torch.tensor(times_norm, device=device, dtype=torch.float32)
    n_images = len(image_paths)
    warmup_steps = int(train_steps * warmup_frac)
    ssim_weight = 0.2

    for step in range(train_steps):
        idx = int(torch.randint(0, n_images, (1,)).item())
        gt_image = images_gt[idx]
        viewmat = viewmats[idx][None]
        K = Ks[idx][None]
        img_h, img_w = gt_image.shape[:2]
        sh_degree_to_use = min(step // 1000, sh_degree)
        use_deform = step >= warmup_steps

        if use_deform:
            means, quats, scales, d_xyz = _apply_deform(splats, deform, times_t[idx])
        else:
            means = splats["means"]
            quats = F.normalize(splats["quats"], dim=-1)
            scales = torch.exp(splats["scales"])
            d_xyz = None

        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        renders, _alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors_sh,
            viewmats=viewmat,
            Ks=K,
            width=img_w,
            height=img_h,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
            packed=False,
            absgrad=True,
        )
        rendered = renders[0]

        densify = (not use_deform) and len(splats["means"]) < max_gaussians
        if densify:
            strategy.step_pre_backward(
                params=splats,
                optimizers=splat_optimizers,
                state=strategy_state,
                step=step,
                info=info,
            )

        l1_loss = F.l1_loss(rendered, gt_image)
        loss = (1 - ssim_weight) * l1_loss + ssim_weight * (1 - _ssim(rendered, gt_image))
        if d_xyz is not None:
            loss = loss + 0.01 * d_xyz.abs().mean()
        loss.backward()

        if densify:
            strategy.step_post_backward(
                params=splats,
                optimizers=splat_optimizers,
                state=strategy_state,
                step=step,
                info=info,
                packed=False,
            )

        for opt in (*splat_optimizers.values(), deform_opt):
            opt.step()
            opt.zero_grad(set_to_none=True)
        lr_scheduler.step()

        if step % 500 == 0 or step == train_steps - 1:
            phase = "deform" if use_deform else "warmup"
            on_progress(
                f"Step {step}/{train_steps} ({phase}) · "
                f"loss={loss.item():.4f} · {len(splats['means'])} Gaussians"
            )

    with torch.no_grad():
        export_splats(
            means=splats["means"].detach(),
            scales=splats["scales"].detach(),
            quats=splats["quats"].detach(),
            opacities=splats["opacities"].detach(),
            sh0=splats["sh0"].detach(),
            shN=splats["shN"].detach(),
            format="ply",
            save_to=str(output_dir / "canonical.ply"),
        )
        torch.save(
            {
                "splats": {k: v.detach().cpu() for k, v in splats.items()},
                "deform": {k: v.detach().cpu() for k, v in deform.state_dict().items()},
                "n_gaussians": len(splats["means"]),
            },
            output_dir / "checkpoint.pt",
        )

    return splats, deform, int(len(splats["means"]))


def _export_time_plys(
    splats: nn.ParameterDict,
    deform: DeformField,
    output_dir: Path,
    *,
    n_times: int,
    duration_sec: float,
) -> dict[str, Any]:
    from gsplat import export_splats

    times_dir = output_dir / "times"
    if times_dir.exists():
        shutil.rmtree(times_dir)
    times_dir.mkdir(parents=True)

    items = []
    ts = export_times(n_times)
    deform.eval()
    with torch.no_grad():
        for i, t in enumerate(ts):
            t_tensor = torch.tensor(t, device=splats["means"].device, dtype=torch.float32)
            means, quats, scales, _ = _apply_deform(splats, deform, t_tensor)
            rel = f"times/t{i:03d}.ply"
            export_splats(
                means=means.detach(),
                scales=torch.log(scales.clamp(min=1e-8)).detach(),
                quats=quats.detach(),
                opacities=splats["opacities"].detach(),
                sh0=splats["sh0"].detach(),
                shN=splats["shN"].detach(),
                format="ply",
                save_to=str(output_dir / rel),
            )
            items.append(
                {
                    "index": i,
                    "t": round(float(t), 4),
                    "t_sec": round(float(t) * duration_sec, 3),
                    "file": rel,
                }
            )

    mid = items[len(items) // 2]
    src = output_dir / mid["file"]
    if src.exists():
        shutil.copy2(src, output_dir / "gaussians.ply")
        shutil.copy2(src, output_dir / "scene.ply")

    return {
        "duration_sec": duration_sec,
        "n_gaussians": int(len(splats["means"])),
        "times": items,
    }
