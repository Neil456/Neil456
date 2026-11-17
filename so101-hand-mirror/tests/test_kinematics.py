import math

import numpy as np
import pytest

from hand_mirror.kinematics import ArmGeometry, JointAngles, JointLimits, Pose, forward, inverse

GEOM = ArmGeometry()


def test_l_pose_forward():
    p = forward(GEOM, JointAngles())  # defaults = L pose
    # Elbow directly above the shoulder, gripper horizontal pointing forward.
    np.testing.assert_allclose(p.elbow[:2], p.shoulder[:2], atol=1e-9)
    assert p.elbow[2] == pytest.approx(GEOM.base_height + GEOM.upper_arm)
    assert p.tip[2] == pytest.approx(p.elbow[2])
    assert p.tip[0] == pytest.approx(GEOM.shoulder_offset + GEOM.forearm + GEOM.gripper)
    assert p.pitch == pytest.approx(0.0)


def test_l_pose_inverse():
    p = forward(GEOM, JointAngles())
    ik = inverse(GEOM, Pose(*p.tip, pitch=0.0))
    assert ik.reachable
    for name, want in JointAngles().as_dict().items():
        assert getattr(ik.q, name) == pytest.approx(want, abs=1e-6), name


@pytest.mark.parametrize("seed", range(5))
def test_round_trip_random(seed):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        q = JointAngles(
            pan=rng.uniform(-90, 90),
            shoulder=rng.uniform(20, 160),
            elbow=rng.uniform(-130, -15),  # elbow-up branch, inside the IK reach margins
            wrist_pitch=rng.uniform(-90, 90),
            wrist_roll=rng.uniform(-90, 90),
        )
        fk = forward(GEOM, q)
        pan_dir = np.array([math.cos(math.radians(q.pan)), math.sin(math.radians(q.pan))])
        if fk.tip[:2] @ pan_dir < 0.02:
            continue  # tip behind / above the base: IK always faces the target instead
        ik = inverse(GEOM, Pose(*fk.tip, pitch=fk.pitch, roll=q.wrist_roll), prev_pan=q.pan)
        fk2 = forward(GEOM, ik.q)
        np.testing.assert_allclose(fk2.tip, fk.tip, atol=1e-6)
        assert ((fk2.pitch - fk.pitch + 180) % 360 - 180) == pytest.approx(0, abs=1e-6)


def test_unreachable_is_pulled_in():
    ik = inverse(GEOM, Pose(1.0, 0.0, 0.1, pitch=0.0))
    assert not ik.reachable
    fk = forward(GEOM, ik.q)
    assert fk.tip[0] < 1.0 and np.all(np.isfinite(fk.tip))


def test_limits_clamp():
    q, clamped = JointLimits().clamp(JointAngles(pan=500))
    assert clamped and q.pan == 110


def test_home_pose_is_comfortable():
    from hand_mirror.mapping import MappingConfig
    h = MappingConfig().home
    ik = inverse(GEOM, Pose(h["x"], h["y"], h["z"], h["pitch"]))
    assert ik.reachable
    _, clamped = JointLimits().clamp(ik.q)
    assert not clamped
    assert 60 < ik.q.shoulder < 120  # upper arm roughly upright
