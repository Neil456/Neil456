"""Forward / inverse kinematics for the SO-101 arm.

The SO-101 is a 5-DOF arm (+ gripper). Joints 2-4 (shoulder lift, elbow, wrist
flex) all rotate about parallel horizontal axes, so once the base has turned to
face the target, the rest of the problem is a *planar* 3-link arm. That gives a
closed-form (analytic) IK solution that is exact, fast and has no iterations.

Frames and conventions used everywhere in this project ("model" angles)
----------------------------------------------------------------------
World frame: origin on the table under the base rotation axis.
    +x  forward (the direction the arm points when pan = 0)
    +y  to the robot's left
    +z  up

All angles are in degrees.
    pan          base yaw, + = turn left (counter-clockwise seen from above)
    shoulder     absolute angle of the upper arm, measured from horizontal-forward.
                 90 = upper arm straight up, >90 = leaning backward.
    elbow        forearm angle *relative* to the upper arm. -90 = forearm at a
                 right angle, bent forward/down (the "L" pose).
    wrist_pitch  gripper angle relative to the forearm, + = tip up.
    wrist_roll   rotation of the gripper about its own axis.

The "L pose" (upper arm vertical, forearm horizontal pointing forward, gripper in
line with the forearm) is therefore: pan=0, shoulder=90, elbow=-90, wrist_pitch=0.
That is the reference pose you put the real arm in when you run
`python -m hand_mirror.calibrate_pose`.

The absolute pitch of the gripper (relative to horizontal) is
    phi = shoulder + elbow + wrist_pitch
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArmGeometry:
    """Link lengths in metres. Defaults are taken from the SO-101 URDF; measure
    your own arm with a ruler (axis centre to axis centre) for best accuracy."""

    base_height: float = 0.117      # table surface -> shoulder-lift axis
    shoulder_offset: float = 0.030  # horizontal distance pan axis -> shoulder-lift axis
    upper_arm: float = 0.116        # shoulder-lift axis -> elbow axis
    forearm: float = 0.135          # elbow axis -> wrist-flex axis
    gripper: float = 0.100          # wrist-flex axis -> point between the jaw tips


@dataclass
class JointAngles:
    pan: float = 0.0
    shoulder: float = 90.0
    elbow: float = -90.0
    wrist_pitch: float = 0.0
    wrist_roll: float = 0.0

    NAMES = ("pan", "shoulder", "elbow", "wrist_pitch", "wrist_roll")

    def as_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in self.NAMES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "JointAngles":
        return cls(**{n: float(d[n]) for n in cls.NAMES})


@dataclass
class JointLimits:
    """Software joint limits in model degrees (the servos also enforce the range
    you recorded with `lerobot-calibrate`)."""

    pan: tuple[float, float] = (-110.0, 110.0)
    shoulder: tuple[float, float] = (5.0, 160.0)
    elbow: tuple[float, float] = (-170.0, 10.0)
    wrist_pitch: tuple[float, float] = (-100.0, 100.0)
    wrist_roll: tuple[float, float] = (-160.0, 160.0)

    def clamp(self, q: JointAngles) -> tuple[JointAngles, bool]:
        """Return (clamped angles, whether anything was clamped)."""
        out = {}
        clamped = False
        for name in JointAngles.NAMES:
            lo, hi = getattr(self, name)
            v = getattr(q, name)
            c = min(max(v, lo), hi)
            clamped |= c != v
            out[name] = c
        return JointAngles(**out), clamped


@dataclass
class Pose:
    """Gripper target: tip position (m), absolute pitch from horizontal (deg,
    + = tip up), and roll (deg)."""

    x: float
    y: float
    z: float
    pitch: float = 0.0
    roll: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.pitch, self.roll], dtype=float)


@dataclass
class ArmPoints:
    """3D positions of each joint, for drawing and tests."""

    base: np.ndarray
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    tip: np.ndarray
    pitch: float = 0.0
    extra: dict = field(default_factory=dict)


def forward(geom: ArmGeometry, q: JointAngles) -> ArmPoints:
    """Joint angles -> 3D points of the arm (tip is the grasp point)."""
    pan = math.radians(q.pan)
    a1 = math.radians(q.shoulder)
    a2 = a1 + math.radians(q.elbow)
    a3 = a2 + math.radians(q.wrist_pitch)

    # Work in the vertical plane (r = radial distance from pan axis, z = height)
    r0, z0 = geom.shoulder_offset, geom.base_height
    r1, z1 = r0 + geom.upper_arm * math.cos(a1), z0 + geom.upper_arm * math.sin(a1)
    r2, z2 = r1 + geom.forearm * math.cos(a2), z1 + geom.forearm * math.sin(a2)
    r3, z3 = r2 + geom.gripper * math.cos(a3), z2 + geom.gripper * math.sin(a3)

    c, s = math.cos(pan), math.sin(pan)

    def to3d(r: float, z: float) -> np.ndarray:
        return np.array([r * c, r * s, z])

    return ArmPoints(
        base=np.zeros(3),
        shoulder=to3d(r0, z0),
        elbow=to3d(r1, z1),
        wrist=to3d(r2, z2),
        tip=to3d(r3, z3),
        pitch=math.degrees(a3),
    )


@dataclass
class IKResult:
    q: JointAngles
    reachable: bool  # False if the target had to be pulled back into the workspace


def inverse(geom: ArmGeometry, target: Pose, prev_pan: float = 0.0) -> IKResult:
    """Analytic IK. Always returns a solution; if the target is out of reach the
    wrist centre is pulled onto the edge of the workspace and `reachable=False`.

    Uses the "elbow up" solution (elbow above the shoulder->wrist line), which is
    the natural posture for the SO-101 and keeps the elbow away from the table.
    """
    L1, L2, L3 = geom.upper_arm, geom.forearm, geom.gripper

    # 1) Base yaw: face the target.
    radial = math.hypot(target.x, target.y)
    pan = math.degrees(math.atan2(target.y, target.x)) if radial > 1e-4 else prev_pan

    # 2) Planar problem, relative to the shoulder-lift axis.
    r = radial - geom.shoulder_offset
    h = target.z - geom.base_height
    phi = math.radians(target.pitch)

    # Wrist-flex axis position = tip pulled back along the gripper direction.
    wr = r - L3 * math.cos(phi)
    wz = h - L3 * math.sin(phi)

    d = math.hypot(wr, wz)
    d_max = (L1 + L2) * 0.995
    # Don't let the arm fold up completely (ugly, near-self-collision postures).
    d_min = max(abs(L1 - L2) * 1.05, 0.35 * (L1 + L2))
    reachable = True
    if d > d_max or d < d_min:
        reachable = False
        scale = (d_max if d > d_max else d_min) / max(d, 1e-9)
        if d < 1e-9:  # degenerate: point straight up
            wr, wz = 0.0, d_min
        else:
            wr, wz = wr * scale, wz * scale
        d = math.hypot(wr, wz)

    # 3) Two-link IK (law of cosines) for shoulder + elbow.
    cos_e = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_e = max(-1.0, min(1.0, cos_e))
    elbow = -math.acos(cos_e)  # negative = elbow up
    shoulder = math.atan2(wz, wr) - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))

    # 4) Wrist takes up whatever pitch is left over.
    wrist = phi - shoulder - elbow

    q = JointAngles(
        pan=pan,
        shoulder=math.degrees(shoulder),
        elbow=math.degrees(elbow),
        wrist_pitch=_wrap180(math.degrees(wrist)),
        wrist_roll=target.roll,
    )
    return IKResult(q=q, reachable=reachable)


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
