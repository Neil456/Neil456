"""Side + top view of the arm, drawn with OpenCV next to the webcam image."""

from __future__ import annotations

import cv2
import numpy as np

from .kinematics import ArmGeometry, ArmPoints, JointAngles, Pose, forward

BG = (30, 30, 34)
GRID = (55, 55, 62)
LINK = (230, 180, 60)
JOINT = (250, 250, 250)
TIP = (0, 200, 255)
TARGET_OK = (80, 220, 80)
TARGET_BAD = (60, 60, 240)
TEXT = (210, 210, 210)


def draw_arm_panel(
    geom: ArmGeometry,
    q: JointAngles,
    target: Pose | None,
    reachable: bool,
    size: tuple[int, int] = (360, 480),
) -> np.ndarray:
    """Returns a BGR image (h, w) = size with a side view on top and a top view below."""
    w, h = size
    panel = np.full((h, w, 3), BG, dtype=np.uint8)
    pts: ArmPoints = forward(geom, q)
    scale = (w - 40) / 0.42  # pixels per metre (~42 cm across)

    # --- Side view: horizontal = radial distance r, vertical = z -------------
    side_h = h // 2
    ox, oy = 40, side_h - 25  # pixel position of (r=0, z=0)
    cv2.putText(panel, "side view", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.line(panel, (0, oy), (w, oy), GRID, 2)  # table

    def side(p: np.ndarray) -> tuple[int, int]:
        r = float(np.hypot(p[0], p[1]))
        return int(ox + r * scale), int(oy - p[2] * scale)

    chain = [pts.base, pts.shoulder, pts.elbow, pts.wrist, pts.tip]
    _draw_chain(panel, [side(p) for p in chain])
    if target is not None:
        tr = np.hypot(target.x, target.y)
        tp = (int(ox + tr * scale), int(oy - target.z * scale))
        _draw_target(panel, tp, target.pitch, reachable)

    # --- Top view: right = +x (forward), up = +y (robot's left) --------------
    cv2.line(panel, (0, side_h), (w, side_h), GRID, 1)
    tox, toy = 40, side_h + (h - side_h) // 2
    cv2.putText(panel, "top view", (8, side_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.circle(panel, (tox, toy), int(0.05 * scale), GRID, 1)

    def top(p: np.ndarray) -> tuple[int, int]:
        return int(tox + p[0] * scale), int(toy - p[1] * scale)

    _draw_chain(panel, [top(p) for p in chain])
    if target is not None:
        cv2.circle(panel, top(np.array([target.x, target.y, 0])), 6, TARGET_OK if reachable else TARGET_BAD, 2)

    lines = [
        f"pan {q.pan:6.1f}  sh {q.shoulder:6.1f}",
        f"elb {q.elbow:6.1f}  wp {q.wrist_pitch:6.1f}",
        f"roll {q.wrist_roll:6.1f}",
    ]
    for i, s in enumerate(lines):
        cv2.putText(panel, s, (8, h - 50 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)
    return panel


def _draw_chain(img, pts):
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, LINK, 5, cv2.LINE_AA)
    for p in pts[:-1]:
        cv2.circle(img, p, 5, JOINT, -1, cv2.LINE_AA)
    cv2.circle(img, pts[-1], 6, TIP, -1, cv2.LINE_AA)


def _draw_target(img, p, pitch_deg, ok):
    color = TARGET_OK if ok else TARGET_BAD
    cv2.circle(img, p, 8, color, 2, cv2.LINE_AA)
    a = np.radians(pitch_deg)
    tail = (int(p[0] - 25 * np.cos(a)), int(p[1] + 25 * np.sin(a)))
    cv2.line(img, tail, p, color, 1, cv2.LINE_AA)
