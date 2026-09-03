from scene_gen.orbit_views import (
    angular_distance,
    camera_xyz,
    describe_azimuth,
    generation_order,
    iter_angles,
    select_context,
)


def test_iter_angles_ten_degree_circle():
    angles = iter_angles(0, 360, 10)
    assert angles[0] == 0
    assert angles[-1] == 350
    assert len(angles) == 36
    assert 360 not in angles


def test_iter_angles_rejects_too_many():
    try:
        iter_angles(0, 360, 1)
    except ValueError as exc:
        assert "72" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_camera_xyz_front_and_side():
    x0, y0, z0 = camera_xyz(0, 0, 2)
    assert abs(x0) < 1e-6 and abs(y0) < 1e-6 and abs(z0 - 2) < 1e-6
    x90, _, z90 = camera_xyz(90, 0, 2)
    assert abs(x90 - 2) < 1e-6 and abs(z90) < 1e-6


def test_describe_azimuth():
    assert "front" in describe_azimuth(0)
    assert "behind" in describe_azimuth(180)


def test_generation_order_bidirectional():
    angles = iter_angles(0, 360, 10)
    order = generation_order(angles, "bidirectional")
    assert order[:5] == [0, 10, 350, 20, 340]
    assert generation_order(angles, "sequential") == angles


def test_select_context_nearest_keeps_original():
    generated = [
        {"angle": 0, "image": "orig"},
        {"angle": 10, "image": "a"},
        {"angle": 350, "image": "b"},
        {"angle": 20, "image": "c"},
    ]
    refs = select_context(30, generated, mode="nearest", max_neighbors=2)
    assert refs[0]["angle"] == 0
    assert [r["angle"] for r in refs[1:]] == [20, 10]


def test_angular_distance_wraps():
    assert angular_distance(350, 10) == 20
