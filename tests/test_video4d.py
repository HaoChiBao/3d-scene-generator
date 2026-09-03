from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene_gen.frames import normalize_times, prepare_timed_folder
from scene_gen.video4d import export_times, scale_intrinsics


def test_normalize_times_unit_interval():
    assert normalize_times([]) == []
    assert normalize_times([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]
    scaled = normalize_times([2.0, 4.0, 6.0])
    assert scaled[0] == 0.0
    assert scaled[-1] == 1.0
    assert abs(scaled[1] - 0.5) < 1e-6


def test_prepare_timed_folder_from_images(tmp_path: Path):
    folder = tmp_path / "shots"
    folder.mkdir()
    for i, color in enumerate([(10, 20, 30), (40, 50, 60), (70, 80, 90)]):
        Image.new("RGB", (32, 24), color=color).save(folder / f"{i:02d}.png")
    work = tmp_path / "work"
    paths, times = prepare_timed_folder(folder, work, max_frames=8)
    assert len(paths) == 3
    assert times == [0.0, 1.0, 2.0]
    assert all(p.exists() for p in paths)


def test_prepare_timed_folder_rejects_still(tmp_path: Path):
    still = tmp_path / "one.jpg"
    Image.new("RGB", (16, 16), color=(8, 8, 8)).save(still)
    try:
        prepare_timed_folder(still, tmp_path / "work")
    except ValueError as exc:
        assert "Video 4D" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_export_times_inclusive():
    assert export_times(1) == [0.0]
    eight = export_times(8)
    assert eight[0] == 0.0
    assert eight[-1] == 1.0
    assert len(eight) == 8


def test_scale_intrinsics_doubles_when_image_grows():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]])
    out = scale_intrinsics(K, (40, 50), (80, 100))
    assert abs(out[0, 0] - 200.0) < 1e-4
    assert abs(out[1, 1] - 160.0) < 1e-4
    assert abs(out[0, 2] - 100.0) < 1e-4
    assert abs(out[1, 2] - 80.0) < 1e-4


def test_positional_encoding_shape():
    torch = pytest.importorskip("torch")
    from scene_gen.video4d import positional_encoding

    x = torch.zeros(5, 3)
    encoded = positional_encoding(x, 4)
    assert encoded.shape == (5, 3 + 8 * 3)


def test_deform_field_starts_at_identity():
    torch = pytest.importorskip("torch")
    from scene_gen.video4d import DeformField

    net = DeformField()
    xyz = torch.randn(16, 3)
    t = torch.full((16, 1), 0.4)
    d_xyz, d_quat, d_scale = net(xyz, t)
    assert d_xyz.abs().max().item() < 1e-6
    assert d_quat.abs().max().item() < 1e-6
    assert d_scale.abs().max().item() < 1e-6
