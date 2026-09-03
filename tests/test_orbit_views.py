from scene_gen.orbit_views import camera_xyz, describe_azimuth, iter_angles


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
