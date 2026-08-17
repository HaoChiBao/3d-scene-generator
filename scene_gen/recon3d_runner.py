"""recon3d wrapper: VGGT poses + per-scene gsplat → navigable 3DGS PLY."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable


ProgressFn = Callable[[str], None]


def reconstruct_with_recon3d(
    media_path: Path,
    output_dir: Path,
    *,
    max_frames: int = 48,
    target_fps: float = 2.0,
    resize_long_edge: int = 960,
    train_steps: int = 7000,
    metric_align: bool = True,
    use_factor_graph: bool = False,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run recon3d and normalize outputs for the API (gaussians.ply + meta.json)."""
    from recon3d.gaussian_train import TrainConfig
    from recon3d.pipeline import PipelineConfig, reconstruct

    def progress(msg: str) -> None:
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    progress("Extracting frames and estimating cameras (VGGT)…")

    if metric_align:
        try:
            import moge  # noqa: F401
        except Exception:  # noqa: BLE001
            progress("MoGe not available — skipping metric alignment")
            metric_align = False

    config = PipelineConfig(
        max_frames=int(max_frames),
        target_fps=float(target_fps),
        resize_long_edge=int(resize_long_edge),
        pose_method="vggt",
        metric_align=bool(metric_align),
        use_factor_graph=bool(use_factor_graph),
        export_mesh=False,
        launch_viewer=False,
        device="cuda",
        train_config=TrainConfig(max_steps=int(train_steps)),
    )

    progress(
        f"Training 3D Gaussians ({train_steps} steps) — usually a few minutes…"
    )
    ply_path = reconstruct(str(media_path), str(output_dir), config)
    ply_path = Path(ply_path)

    # recon3d's export_all() overwrites scene.ply with a colored *point cloud*.
    # Restore a real 3DGS PLY from the training checkpoint for the web viewer.
    gaussians_ply = output_dir / "gaussians.ply"
    points_ply = output_dir / "points.ply"
    scene_ply = output_dir / "scene.ply"
    if scene_ply.exists() and _is_point_cloud_ply(scene_ply):
        shutil.copy2(scene_ply, points_ply)
    _export_gaussians_from_checkpoint(output_dir / "checkpoint.pt", gaussians_ply)

    if not gaussians_ply.exists():
        # Fallback: keep whatever reconstruct returned (may be points-only).
        if ply_path.exists():
            shutil.copy2(ply_path, gaussians_ply)
        elif scene_ply.exists():
            shutil.copy2(scene_ply, gaussians_ply)

    if not gaussians_ply.exists():
        raise RuntimeError(f"recon3d finished but no PLY found under {output_dir}")

    n_gaussians = _count_gaussians(output_dir / "checkpoint.pt")
    elapsed = time.time() - t0
    meta = {
        "model": "recon3d",
        "pipeline": "VGGT → gsplat",
        "has_gaussians": _is_gaussian_ply(gaussians_ply),
        "has_points": points_ply.exists(),
        "n_gaussians": n_gaussians,
        "max_frames": max_frames,
        "target_fps": target_fps,
        "resize_long_edge": resize_long_edge,
        "train_steps": train_steps,
        "metric_align": metric_align,
        "elapsed_sec": round(elapsed, 1),
        "artifacts": {
            "gaussians": "gaussians.ply",
            "points": "points.ply" if points_ply.exists() else None,
            "scene": "scene.ply" if scene_ply.exists() else None,
            "splat": "scene.splat" if (output_dir / "scene.splat").exists() else None,
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    progress(f"Done — {n_gaussians or '?'} Gaussians in {elapsed:.0f}s")
    return meta


def _count_gaussians(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return int(ckpt.get("n_gaussians", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _ply_header(path: Path, max_bytes: int = 4096) -> str:
    raw = path.read_bytes()[:max_bytes]
    text = raw.decode("latin1", errors="ignore")
    end = text.find("end_header")
    return text if end < 0 else text[: end + len("end_header")]


def _is_point_cloud_ply(path: Path) -> bool:
    header = _ply_header(path).lower()
    return (
        "property uchar red" in header
        and "opacity" not in header
        and "scale_0" not in header
    )


def _is_gaussian_ply(path: Path) -> bool:
    if not path.exists():
        return False
    header = _ply_header(path).lower()
    return "opacity" in header or "scale_0" in header or "f_dc_0" in header


def _export_gaussians_from_checkpoint(checkpoint_path: Path, out_ply: Path) -> None:
    """Rewrite a viewer-compatible 3DGS PLY from recon3d's checkpoint."""
    if not checkpoint_path.exists():
        return
    import torch
    from gsplat import export_splats

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    splats = ckpt.get("splats")
    if not splats:
        return

    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=str(out_ply),
    )
    # Also restore scene.ply as the true Gaussian asset for downstream tools.
    if out_ply.exists():
        shutil.copy2(out_ply, out_ply.parent / "scene.ply")
