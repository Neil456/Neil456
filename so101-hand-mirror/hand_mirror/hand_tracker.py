"""Webcam hand tracking with MediaPipe Hand Landmarker.

MediaPipe gives 21 landmarks per hand, each in two forms:
  * image landmarks: x, y normalised to [0, 1] by image width / height
  * world landmarks: metric 3D (metres) relative to the hand's own centre

From those we derive the handful of signals that drive the arm:
  center  where the palm is in the image (-> robot left/right + up/down)
  depth   how far the hand is from the camera (-> robot forward/back)
  pitch   fingers tilted up / down (-> gripper pitch)
  roll    hand rotation in the image plane (-> wrist roll)
  pinch   thumb-index distance (-> gripper open/close)

Landmark indices: 0 wrist, 4 thumb tip, 5 index MCP, 8 index tip,
9 middle MCP, 13 ring MCP, 17 pinky MCP.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 13, 17
PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
# Rigid palm segments used to estimate distance from the camera.
PALM_SEGMENTS = [(WRIST, INDEX_MCP), (WRIST, PINKY_MCP), (INDEX_MCP, PINKY_MCP), (WRIST, MIDDLE_MCP)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


@dataclass
class HandObservation:
    image_lm: np.ndarray   # (21, 3) normalised image coords
    world_lm: np.ndarray   # (21, 3) metres, hand-centred
    handedness: str        # "Left" / "Right" (correct for the mirrored/selfie image)
    score: float

    center: tuple[float, float]  # palm centre, normalised [0, 1] (x right, y down)
    depth: float                 # proportional to distance from camera (unitless)
    pitch: float                 # deg, + = fingers pointing up
    roll: float                  # deg, in-image rotation of the knuckle line
    pinch: float                 # thumb-index distance / palm length (~0.1 closed, ~1+ open)


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        prefer_hand: str = "any",
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        # Imported here so the rest of the package (IK, tests) works without mediapipe.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(Path(model_path)))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.prefer_hand = prefer_hand.capitalize()
        self._last_ts = -1

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        """Run detection on a (mirrored / selfie-view) BGR frame."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        # Pick a hand: the preferred side if requested, otherwise the most confident.
        candidates = []
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0]
            candidates.append((cat.category_name, cat.score, lms, result.hand_world_landmarks[i]))
        if self.prefer_hand in ("Left", "Right"):
            preferred = [c for c in candidates if c[0] == self.prefer_hand]
            if not preferred:
                return None
            candidates = preferred
        side, score, lms, wlms = max(candidates, key=lambda c: c[1])

        image_lm = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
        world_lm = np.array([[p.x, p.y, p.z] for p in wlms], dtype=float)
        return compute_observation(image_lm, world_lm, w, h, side, score)


def compute_observation(
    image_lm: np.ndarray, world_lm: np.ndarray, width: int, height: int, side: str = "Right", score: float = 1.0
) -> HandObservation:
    """Turn raw landmarks into control signals. Pure function -> easy to test."""
    # Image coords with equal scale on both axes (units of image width).
    px = image_lm[:, :2] * np.array([1.0, height / width])

    center_px = px[PALM].mean(axis=0)
    center = (float(center_px[0]), float(center_px[1] * width / height))

    # Depth via the pinhole model: image_size = f * true_size / distance.
    # The world landmarks tell us the true size of each palm segment *as projected
    # onto the image plane* (so tilting the hand doesn't fool us).
    world_len = sum(np.linalg.norm(world_lm[a, :2] - world_lm[b, :2]) for a, b in PALM_SEGMENTS)
    image_len = sum(np.linalg.norm(px[a] - px[b]) for a, b in PALM_SEGMENTS)
    depth = float(world_len / max(image_len, 1e-6))

    # Pitch: direction wrist -> middle knuckle, angle above horizontal.
    v = world_lm[MIDDLE_MCP] - world_lm[WRIST]  # x right, y down, z away from camera
    pitch = math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2])))

    # Roll: angle of the knuckle line (index MCP -> pinky MCP) in the image.
    k = px[PINKY_MCP] - px[INDEX_MCP]
    roll = math.degrees(math.atan2(k[1], k[0]))

    palm_len = np.linalg.norm(world_lm[MIDDLE_MCP] - world_lm[WRIST])
    pinch = float(np.linalg.norm(world_lm[THUMB_TIP] - world_lm[INDEX_TIP]) / max(palm_len, 1e-6))

    return HandObservation(
        image_lm=image_lm,
        world_lm=world_lm,
        handedness=side,
        score=float(score),
        center=center,
        depth=depth,
        pitch=pitch,
        roll=roll,
        pinch=pinch,
    )


def draw_hand(frame: np.ndarray, obs: HandObservation, color=(0, 255, 120)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in obs.image_lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], (0, 200, 255), 2, cv2.LINE_AA)
    cx, cy = int(obs.center[0] * w), int(obs.center[1] * h)
    cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
