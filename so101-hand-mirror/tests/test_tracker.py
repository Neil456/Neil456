"""Runs MediaPipe on a real photo of a hand (downloaded on first run).
Skipped if mediapipe isn't installed or there is no internet."""
import urllib.request
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from hand_mirror.hand_tracker import HandTracker, ensure_model  # noqa: E402

IMG = Path(__file__).parent / "data" / "hand.jpg"
IMG_URL = "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"


@pytest.fixture(scope="module")
def photo():
    try:
        ensure_model()
        if not IMG.exists():
            IMG.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(IMG_URL, IMG)
    except OSError as e:
        pytest.skip(f"could not download test assets: {e}")
    return cv2.imread(str(IMG))


def test_detects_hand(photo):
    tracker = HandTracker()
    obs = None
    for i in range(3):
        obs = tracker.process(photo, i * 33)
    tracker.close()
    assert obs is not None
    assert 0 <= obs.center[0] <= 1 and 0 <= obs.center[1] <= 1
    assert obs.depth > 0 and obs.pinch > 0
