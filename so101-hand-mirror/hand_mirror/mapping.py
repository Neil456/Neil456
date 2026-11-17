"""Hand observation -> gripper target pose.

Control is *relative*: when you press `space`, wherever your hand is becomes the
"neutral" pose and maps to the arm's home pose. Moving away from neutral moves
the gripper by a proportional amount. This avoids having to know the exact
camera position, your arm length, etc.

Default ("mirror") behaviour, robot placed next to the webcam facing you:
    hand right / left (your view) -> gripper moves to the same side, like a mirror
    hand up / down                -> gripper up / down
    hand toward camera            -> gripper reaches toward you
    tilt fingers down / up        -> gripper pitches down / up
    rotate hand                   -> wrist roll
    pinch thumb + index           -> close gripper
Every axis has a sign in config.yaml so you can flip anything that feels backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hand_tracker import HandObservation
from .kinematics import Pose


@dataclass
class MappingConfig:
    home: dict = field(default_factory=lambda: {"x": 0.24, "y": 0.0, "z": 0.14, "pitch": -20.0, "roll": 0.0})

    # metres of robot motion per unit of hand motion
    lateral_gain: float = 0.45   # per full image width
    vertical_gain: float = 0.40  # per full image height
    depth_gain: float = 0.35     # per 100% change in distance from camera
    pitch_gain: float = 1.0      # deg per deg
    roll_gain: float = 1.0       # deg per deg

    lateral_sign: float = 1.0
    vertical_sign: float = 1.0
    depth_sign: float = 1.0
    pitch_sign: float = 1.0
    roll_sign: float = 1.0

    track_pitch: bool = True
    track_roll: bool = True

    # Workspace box for the gripper tip (metres) - keeps it off the table.
    x_range: tuple[float, float] = (0.10, 0.34)
    y_range: tuple[float, float] = (-0.25, 0.25)
    z_range: tuple[float, float] = (0.03, 0.30)
    pitch_range: tuple[float, float] = (-80.0, 60.0)
    roll_range: tuple[float, float] = (-120.0, 120.0)

    # Gripper: pinch ratio (thumb-index distance / palm length) -> gripper command
    pinch_closed: float = 0.25
    pinch_open: float = 0.90
    gripper_closed: float = 2.0    # LeRobot gripper units, 0..100
    gripper_open: float = 60.0


@dataclass
class Neutral:
    u: float
    v: float
    depth: float
    pitch: float
    roll: float


def _clamp(v: float, rng: tuple[float, float]) -> float:
    return min(max(v, rng[0]), rng[1])


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class HandToPose:
    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.neutral: Neutral | None = None

    def set_neutral(self, obs: HandObservation) -> None:
        self.neutral = Neutral(obs.center[0], obs.center[1], obs.depth, obs.pitch, obs.roll)

    def home_pose(self) -> Pose:
        h = self.cfg.home
        return Pose(h["x"], h["y"], h["z"], h["pitch"], h["roll"])

    def __call__(self, obs: HandObservation) -> tuple[Pose, float]:
        """Return (target pose, gripper command 0..100)."""
        if self.neutral is None:
            self.set_neutral(obs)
        c, n, home = self.cfg, self.neutral, self.home_pose()

        du = obs.center[0] - n.u            # + = hand moved right in the selfie view
        dv = obs.center[1] - n.v            # + = hand moved down
        closer = (n.depth - obs.depth) / n.depth  # + = hand moved toward camera

        x = home.x + c.depth_sign * c.depth_gain * closer
        y = home.y + c.lateral_sign * c.lateral_gain * du
        z = home.z - c.vertical_sign * c.vertical_gain * dv

        pitch = home.pitch
        if c.track_pitch:
            pitch += c.pitch_sign * c.pitch_gain * (obs.pitch - n.pitch)
        roll = home.roll
        if c.track_roll:
            roll += c.roll_sign * c.roll_gain * _wrap180(obs.roll - n.roll)

        pose = Pose(
            x=_clamp(x, c.x_range),
            y=_clamp(y, c.y_range),
            z=_clamp(z, c.z_range),
            pitch=_clamp(pitch, c.pitch_range),
            roll=_clamp(roll, c.roll_range),
        )

        t = (obs.pinch - c.pinch_closed) / max(c.pinch_open - c.pinch_closed, 1e-6)
        t = min(max(t, 0.0), 1.0)
        grip = c.gripper_closed + t * (c.gripper_open - c.gripper_closed)
        return pose, grip
