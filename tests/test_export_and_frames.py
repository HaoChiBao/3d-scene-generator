"""CPU-only smoke tests for frame prep and PLY export."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_gen.export_ply import predictions_to_point_cloud, write_ply
from scene_gen.frames import prepare_image_folder


def test_prepare_image_folder(tmp_path: Path):
    img = tmp_path / "shot.png"
    Image.new("RGB", (64, 48), color=(20, 80, 160)).save(img)
    work = tmp_path / "work"
    paths = prepare_image_folder(img, work, max_frames=4)
    assert len(paths) == 1
    assert paths[0].exists()


def test_write_ply_roundtrip(tmp_path: Path):
    predictions = {
        "world_points": np.random.randn(2, 8, 8, 3).astype(np.float32),
        "world_points_conf": np.linspace(0.1, 1.0, 2 * 8 * 8).reshape(2, 8, 8),
        "images": np.clip(np.random.rand(2, 3, 8, 8), 0, 1).astype(np.float32),
    }
    xyz, rgb = predictions_to_point_cloud(predictions, conf_thres=20, max_points=1000)
    out = tmp_path / "scene.ply"
    write_ply(out, xyz, rgb)
    text = out.read_text(encoding="utf-8")
    assert "element vertex" in text
    assert len(xyz) > 0
