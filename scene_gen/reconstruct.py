"""VGGT-based whole-space reconstruction."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scene_gen.export_ply import predictions_to_point_cloud, write_ply
from scene_gen.frames import prepare_image_folder


DEFAULT_VGGT_REPO = Path("/opt/vggt")
DEFAULT_MODEL_REPO = "facebook/VGGT-1B"
DEFAULT_MODEL_FILE = "model.pt"


def ensure_vggt_on_path(vggt_repo: Path = DEFAULT_VGGT_REPO) -> None:
    repo = str(vggt_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_vggt_model(
    device: torch.device,
    *,
    vggt_repo: Path = DEFAULT_VGGT_REPO,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
):
    """Load VGGT weights via Hugging Face Hub (respects HF_HOME / Modal volumes)."""
    ensure_vggt_on_path(vggt_repo)
    from huggingface_hub import hf_hub_download
    from vggt.models.vggt import VGGT

    print(f"Downloading/loading {model_repo}/{model_file} via huggingface_hub...")
    weights_path = hf_hub_download(repo_id=model_repo, filename=model_file)
    print(f"Weights path: {weights_path}")

    model = VGGT()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    return model


@torch.inference_mode()
def run_vggt_on_images(
    model,
    image_paths: list[Path],
    device: torch.device,
) -> dict[str, Any]:
    ensure_vggt_on_path()
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    if device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    else:
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key, value in list(predictions.items()):
        if isinstance(value, torch.Tensor):
            predictions[key] = value.detach().cpu().numpy().squeeze(0)

    predictions["pose_enc_list"] = None
    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    return predictions


def reconstruct_media(
    media_path: Path,
    output_dir: Path,
    model,
    *,
    device: torch.device | None = None,
    target_fps: float = 1.0,
    max_frames: int = 24,
    conf_thres: float = 50.0,
    prediction_mode: str = "Pointmap",
) -> dict[str, Any]:
    """
    End-to-end: media → frames → VGGT → scene.ply + meta.json
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    image_paths = prepare_image_folder(
        media_path,
        work_dir,
        target_fps=target_fps,
        max_frames=max_frames,
    )
    t_frames = time.time()

    predictions = run_vggt_on_images(model, image_paths, device)
    t_infer = time.time()

    vertices, colors = predictions_to_point_cloud(
        predictions,
        conf_thres=conf_thres,
        prediction_mode=prediction_mode,
    )
    ply_path = output_dir / "scene.ply"
    write_ply(ply_path, vertices, colors)

    meta = {
        "model": "facebook/VGGT-1B",
        "num_frames": len(image_paths),
        "num_points": int(len(vertices)),
        "conf_thres": conf_thres,
        "prediction_mode": prediction_mode,
        "device": str(device),
        "timings_sec": {
            "frame_extract": round(t_frames - t0, 3),
            "inference": round(t_infer - t_frames, 3),
            "export": round(time.time() - t_infer, 3),
            "total": round(time.time() - t0, 3),
        },
        "frame_names": [p.name for p in image_paths],
        "bbox": {
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return meta
