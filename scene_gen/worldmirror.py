"""WorldMirror 2.0 high-quality whole-space reconstruction."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_HYWORLD_REPO = Path("/opt/HY-World-2.0")
DEFAULT_MODEL = "tencent/HY-World-2.0"
DEFAULT_SUBFOLDER = "HY-WorldMirror-2.0"


def load_worldmirror_pipeline(
    *,
    model_id: str = DEFAULT_MODEL,
    subfolder: str = DEFAULT_SUBFOLDER,
    enable_bf16: bool = True,
):
    """Load WorldMirror 2.0 pipeline (weights cached under HF_HOME)."""
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    pipeline = WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        subfolder=subfolder,
        enable_bf16=enable_bf16,
    )
    return pipeline


def reconstruct_with_worldmirror(
    media_path: Path,
    output_dir: Path,
    pipeline,
    *,
    target_size: int = 952,
    fps: int = 1,
    video_max_frames: int = 32,
    video_min_frames: int = 1,
) -> dict[str, Any]:
    """
    Run WorldMirror on an image folder or video.

    Writes:
      - gaussians.ply  (primary 3DGS asset)
      - points.ply     (fallback point cloud, if produced)
      - meta.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    work_out = output_dir / "worldmirror_raw"
    if work_out.exists():
        shutil.rmtree(work_out)
    work_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result_dir = pipeline(
        str(media_path),
        output_path=str(work_out),
        strict_output_path=str(work_out),
        target_size=int(target_size),
        fps=int(fps),
        video_max_frames=int(video_max_frames),
        video_min_frames=int(video_min_frames),
        save_gs=True,
        save_points=True,
        save_depth=False,
        save_normal=False,
        save_camera=True,
        save_rendered=False,
        # Avoid onnxruntime sky segmentation dependency in the Modal image.
        apply_sky_mask=False,
        apply_edge_mask=True,
        log_time=True,
    )
    t1 = time.time()

    search_root = Path(result_dir) if result_dir else work_out
    gauss = _find_file(search_root, "gaussians.ply")
    points = _find_file(search_root, "points.ply")
    cameras = _find_file(search_root, "camera_params.json")

    if gauss is None and points is None:
        raise RuntimeError(
            f"WorldMirror produced no gaussians.ply or points.ply under {search_root}"
        )

    if gauss is not None:
        shutil.copy2(gauss, output_dir / "gaussians.ply")
        # Keep legacy filename for older clients; prefer gaussians content.
        shutil.copy2(gauss, output_dir / "scene.ply")
    if points is not None:
        shutil.copy2(points, output_dir / "points.ply")
        if gauss is None:
            shutil.copy2(points, output_dir / "scene.ply")
    if cameras is not None:
        shutil.copy2(cameras, output_dir / "camera_params.json")

    meta = {
        "model": "tencent/HY-World-2.0 (WorldMirror 2.0)",
        "backend": "worldmirror",
        "input": media_path.name,
        "target_size": target_size,
        "fps": fps,
        "video_max_frames": video_max_frames,
        "has_gaussians": gauss is not None,
        "has_points": points is not None,
        "artifacts": {
            "gaussians": "gaussians.ply" if gauss is not None else None,
            "points": "points.ply" if points is not None else None,
            "scene": "scene.ply",
        },
        "timings_sec": {
            "inference": round(t1 - t0, 3),
            "total": round(time.time() - t0, 3),
        },
        "raw_output": str(search_root),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None
