import numpy as np
import pytest

from hand_mirror.hand_tracker import HandObservation
from hand_mirror.kinematics import JointAngles
from hand_mirror.mapping import HandToPose, MappingConfig
from hand_mirror.robot import JointCal, JointMap


def obs(u=0.5, v=0.5, depth=1.0, pitch=80.0, roll=180.0, pinch=1.0):
    z = np.zeros((21, 3))
    return HandObservation(z, z, "Right", 1.0, (u, v), depth, pitch, roll, pinch)


def test_neutral_maps_to_home():
    m = HandToPose(MappingConfig())
    m.set_neutral(obs())
    pose, _ = m(obs())
    home = m.home_pose()
    assert (pose.x, pose.y, pose.z, pose.pitch, pose.roll) == pytest.approx(
        (home.x, home.y, home.z, home.pitch, home.roll))


def test_directions():
    m = HandToPose(MappingConfig())
    m.set_neutral(obs())
    home = m.home_pose()
    assert m(obs(u=0.6))[0].y > home.y        # hand to your right -> robot's left (mirror)
    assert m(obs(v=0.4))[0].z > home.z        # hand up -> gripper up
    assert m(obs(depth=0.9))[0].x > home.x    # hand closer -> reach toward you
    assert m(obs(pitch=60))[0].pitch < home.pitch  # fingers tilt down -> gripper down


def test_roll_wraps_across_180():
    m = HandToPose(MappingConfig())
    m.set_neutral(obs(roll=175))
    assert m(obs(roll=-175))[0].roll == pytest.approx(10)


def test_gripper():
    m = HandToPose(MappingConfig())
    m.set_neutral(obs())
    c = m.cfg
    assert m(obs(pinch=0.05))[1] == pytest.approx(c.gripper_closed)
    assert m(obs(pinch=2.0))[1] == pytest.approx(c.gripper_open)


def test_workspace_clamp():
    m = HandToPose(MappingConfig())
    m.set_neutral(obs())
    pose, _ = m(obs(v=5.0))  # hand way below
    assert pose.z == m.cfg.z_range[0]


def test_joint_map_round_trip(tmp_path):
    jm = JointMap({
        "pan": JointCal(3.0, 0.0, -1.0),
        "shoulder": JointCal(-40.0, 90.0, 1.0),
        "elbow": JointCal(55.0, -90.0, -1.0),
        "wrist_pitch": JointCal(10.0, 0.0, 1.0),
        "wrist_roll": JointCal(0.0, 0.0, -1.0),
    })
    path = tmp_path / "cal.json"
    jm.save(path)
    jm = JointMap.load(path)
    q = JointAngles(10, 100, -70, 20, 30)
    motor = jm.to_motor(q)
    assert motor["shoulder_pan"] == pytest.approx(-7.0)
    assert motor["elbow_flex"] == pytest.approx(55.0 - 20.0)
    back = jm.to_model(motor)
    assert back.as_dict() == pytest.approx(q.as_dict())
