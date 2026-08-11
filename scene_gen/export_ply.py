"""Export VGGT predictions to a colored PLY point cloud."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def predictions_to_point_cloud(
    predictions: dict,
    *,
    conf_thres: float = 50.0,
    prediction_mode: str = "Pointmap",
    max_points: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert VGGT outputs into filtered XYZ + RGB arrays.

    conf_thres is a percentile: keep the top (100 - conf_thres)% confident points.
    """
    if "Pointmap" in prediction_mode and "world_points" in predictions:
        points = predictions["world_points"]
        conf = predictions.get("world_points_conf", np.ones_like(points[..., 0]))
    else:
        points = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf", np.ones_like(points[..., 0]))

    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        colors = np.transpose(images, (0, 2, 3, 1))
    else:
        colors = images

    vertices = points.reshape(-1, 3)
    colors_rgb = (colors.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    finite = np.isfinite(vertices).all(axis=1)
    vertices = vertices[finite]
    colors_rgb = colors_rgb[finite]
    conf_flat = conf_flat[finite]

    if vertices.size == 0:
        raise ValueError("No valid 3D points produced.")

    if conf_thres > 0 and conf_flat.size > 0:
        threshold = np.percentile(conf_flat, conf_thres)
        keep = conf_flat >= threshold
        # Always keep at least some points if percentile is too aggressive
        if keep.sum() < 1000:
            keep = conf_flat >= np.percentile(conf_flat, max(conf_thres - 25, 0))
        vertices = vertices[keep]
        colors_rgb = colors_rgb[keep]

    if len(vertices) > max_points:
        idx = np.random.default_rng(0).choice(len(vertices), size=max_points, replace=False)
        vertices = vertices[idx]
        colors_rgb = colors_rgb[idx]

    return vertices.astype(np.float32), colors_rgb


def write_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY that Three.js PLYLoader can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(vertices)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
