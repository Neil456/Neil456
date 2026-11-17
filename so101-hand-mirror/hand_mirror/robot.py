"""Talking to the SO-101 through LeRobot, plus a fake arm for --dry-run.

LeRobot (with `use_degrees=True`) reports each joint in degrees, where 0 is the
*middle of the range you swept during `lerobot-calibrate`* - not a meaningful
kinematic pose, and the rotation direction depends on how each servo was
mounted. `JointMap` fixes that with one offset + one sign per joint:

    model_angle = model_ref + sign * (motor_deg - motor_ref)

`motor_ref` is what the motor reads when the arm is in the reference "L pose"
(see kinematics.py) and `sign` is +1/-1. Both come from calibrate_pose.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .kinematics import JointAngles

# model joint name -> LeRobot motor name
MOTOR_NAMES = {
    "pan": "shoulder_pan",
    "shoulder": "shoulder_lift",
    "elbow": "elbow_flex",
    "wrist_pitch": "wrist_flex",
    "wrist_roll": "wrist_roll",
}
GRIPPER = "gripper"

# Model angles of the reference "L pose".
REFERENCE_POSE = JointAngles(pan=0.0, shoulder=90.0, elbow=-90.0, wrist_pitch=0.0, wrist_roll=0.0)


@dataclass
class JointCal:
    motor_ref: float
    model_ref: float
    sign: float


class JointMap:
    def __init__(self, cal: dict[str, JointCal]):
        self.cal = cal

    @classmethod
    def load(cls, path: Path | str) -> "JointMap":
        data = json.loads(Path(path).read_text())
        return cls({k: JointCal(**v) for k, v in data["joints"].items()})

    def save(self, path: Path | str) -> None:
        data = {"joints": {k: vars(v) for k, v in self.cal.items()}}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def identity(cls) -> "JointMap":
        """Motor degrees == model degrees. Used by the dry-run arm."""
        return cls({k: JointCal(0.0, 0.0, 1.0) for k in MOTOR_NAMES})

    def to_motor(self, q: JointAngles) -> dict[str, float]:
        out = {}
        for joint, motor in MOTOR_NAMES.items():
            c = self.cal[joint]
            out[motor] = c.motor_ref + c.sign * (getattr(q, joint) - c.model_ref)
        return out

    def to_model(self, motor: dict[str, float]) -> JointAngles:
        vals = {}
        for joint, name in MOTOR_NAMES.items():
            c = self.cal[joint]
            vals[joint] = c.model_ref + c.sign * (motor[name] - c.motor_ref)
        return JointAngles(**vals)


class SO101Arm:
    """Thin wrapper around LeRobot's SO101Follower. Positions in motor degrees
    (gripper 0..100)."""

    def __init__(self, port: str, robot_id: str, max_relative_target: float | None = 25.0):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # older LeRobot releases (<= 0.3)
            try:
                from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            except ImportError as e:
                raise SystemExit(
                    "LeRobot is not installed. Run: pip install 'lerobot[feetech]'\n"
                    "(or use --dry-run to try everything without the arm)"
                ) from e

        cfg = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
            max_relative_target=max_relative_target,
        )
        self.robot = SO101Follower(cfg)

    def connect(self) -> None:
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise SystemExit(
                f"No LeRobot calibration found for robot id '{self.robot.id}'.\n"
                "Run `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> "
                "--robot.id=<id>` first, and pass the same --robot-id here."
            )

    def read(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}

    def write(self, positions: dict[str, float]) -> None:
        self.robot.send_action({f"{k}.pos": float(v) for k, v in positions.items()})

    def torque(self, enabled: bool) -> None:
        if enabled:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()

    def disconnect(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()


class DryRunArm:
    """Pretends to be an arm: remembers the last command and reads it back."""

    def __init__(self):
        q = REFERENCE_POSE
        self.state = {MOTOR_NAMES[k]: v for k, v in q.as_dict().items()}
        self.state[GRIPPER] = 30.0

    def connect(self) -> None:
        pass

    def read(self) -> dict[str, float]:
        return dict(self.state)

    def write(self, positions: dict[str, float]) -> None:
        self.state.update(positions)

    def torque(self, enabled: bool) -> None:
        pass

    def disconnect(self) -> None:
        pass
