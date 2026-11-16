"""Real-time hand mirroring.

    python -m hand_mirror.run --dry-run                       # no robot, just visualise
    python -m hand_mirror.run --port /dev/ttyACM0 --robot-id my_follower

Keys (focus the video window):
    space  start / stop tracking (your current hand pose becomes "neutral")
    c      re-centre: current hand pose becomes neutral
    h      stop tracking and move to the home pose
    p / r  toggle wrist pitch / roll tracking
    q/Esc  quit (arm returns to where it started, then torque is released)

Pipeline, every frame:
    webcam -> MediaPipe landmarks -> hand signals -> target pose (mapping.py)
           -> One Euro smoothing -> analytic IK (kinematics.py) -> joint limits
           -> model->motor degrees (robot.py) -> speed limit -> LeRobot send_action
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .filters import OneEuroFilter, RateLimiter
from .hand_tracker import DEFAULT_MODEL_PATH, HandTracker, draw_hand
from .kinematics import JointAngles, Pose, inverse
from .mapping import HandToPose
from .robot import GRIPPER, DryRunArm, JointMap, SO101Arm
from .viz import draw_arm_panel

DEFAULT_CAL_PATH = Path(__file__).resolve().parent.parent / "calibration.json"
RAMP_UP_S = 2.0  # after engaging, speed ramps from 20% to 100% over this long


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mirror your hand with an SO-101 arm.")
    p.add_argument("--port", help="Serial port of the follower arm, e.g. /dev/ttyACM0 or COM5")
    p.add_argument("--robot-id", default="my_awesome_follower_arm",
                   help="Same id you used with lerobot-calibrate")
    p.add_argument("--dry-run", action="store_true", help="No robot: simulate the arm on screen")
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("--calibration", type=Path, default=DEFAULT_CAL_PATH,
                   help="Joint calibration from calibrate_pose.py")
    p.add_argument("--camera", default="0", help="Camera index or a video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="hand_landmarker.task path")
    p.add_argument("--headless", action="store_true", help="Don't open a window (for testing)")
    p.add_argument("--record", type=Path, help="Save the visualisation to an .mp4")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run forever)")
    p.add_argument("--auto-engage", action="store_true", help="Start tracking on first detected hand")
    return p.parse_args(argv)


class HandMirror:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config)
        c = self.cfg

        if args.dry_run:
            self.arm = DryRunArm()
            self.jmap = JointMap.identity()
        else:
            if not args.port:
                raise SystemExit("--port is required (or use --dry-run)")
            if not args.calibration.exists():
                raise SystemExit(
                    f"{args.calibration} not found. Run `python -m hand_mirror.calibrate_pose "
                    f"--port {args.port} --robot-id {args.robot_id}` first."
                )
            self.arm = SO101Arm(args.port, args.robot_id, c.control.max_relative_target)
            self.jmap = JointMap.load(args.calibration)

        self.tracker = HandTracker(args.model, prefer_hand=c.control.hand_prefer)
        self.mapper = HandToPose(c.mapping)
        self.euro = OneEuroFilter(c.smoothing.min_cutoff, c.smoothing.beta)
        self.limiter = RateLimiter(c.control.max_speed_deg_s)

        self.tracking = False
        self.homing = False
        self.engaged_at = 0.0
        self.last_seen = 0.0
        self.park: dict[str, float] = {}
        self.cmd: dict[str, float] = {}
        self.target: Pose | None = None
        self.reachable = True
        self.status = "press SPACE to start tracking"

    # ------------------------------------------------------------------ robot
    def _speed_scale(self, now: float) -> float:
        return float(np.clip(0.2 + 0.8 * (now - self.engaged_at) / RAMP_UP_S, 0.2, 1.0))

    def _command(self, motor_target: dict[str, float], now: float, scale: float = 1.0) -> None:
        base = self.cfg.control.max_speed_deg_s
        self.limiter.max_rate = {k: v * scale for k, v in base.items()}
        self.cmd = self.limiter(motor_target, now)
        self.arm.write(self.cmd)

    def _pose_to_motor(self, pose: Pose, grip: float) -> dict[str, float]:
        prev_pan = self.jmap.to_model(self.cmd).pan if self.cmd else 0.0
        ik = inverse(self.cfg.geometry, pose, prev_pan=prev_pan)
        q, clamped = self.cfg.limits.clamp(ik.q)
        self.reachable = ik.reachable and not clamped
        motor = self.jmap.to_motor(q)
        motor[GRIPPER] = grip
        return motor

    def _sync_to_robot(self, now: float) -> None:
        """Make the rate limiter start from where the arm really is."""
        current = self.arm.read()
        self.cmd = dict(current)
        self.limiter.reset(current, now)

    # ------------------------------------------------------------------- keys
    def start_tracking(self, obs, now: float) -> None:
        self.mapper.set_neutral(obs)
        self.euro.reset()
        self._sync_to_robot(now)
        self.tracking, self.homing = True, False
        self.engaged_at = self.last_seen = now
        self.status = "TRACKING"

    def stop_tracking(self, why: str) -> None:
        self.tracking = False
        self.status = why

    def handle_key(self, key: int, obs, now: float) -> bool:
        """Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            if self.tracking:
                self.stop_tracking("paused - press SPACE to resume")
            elif obs is not None:
                self.start_tracking(obs, now)
            else:
                self.status = "no hand visible"
        elif key == ord("c") and obs is not None:
            self.mapper.set_neutral(obs)
            self.euro.reset()
            self.status = "re-centred" if not self.tracking else "TRACKING (re-centred)"
        elif key == ord("h"):
            self.stop_tracking("going home")
            self._sync_to_robot(now)
            self.homing, self.engaged_at = True, now
        elif key == ord("p"):
            self.cfg.mapping.track_pitch = not self.cfg.mapping.track_pitch
        elif key == ord("r"):
            self.cfg.mapping.track_roll = not self.cfg.mapping.track_roll
        return True

    # ------------------------------------------------------------------- loop
    def step(self, obs, now: float) -> None:
        if self.tracking:
            if obs is None:
                if now - self.last_seen > self.cfg.control.lost_timeout_s:
                    self.stop_tracking("hand lost - holding. SPACE to resume")
                return  # hold last command
            self.last_seen = now
            pose, grip = self.mapper(obs)
            smoothed = self.euro(np.append(pose.as_array(), grip), now)
            self.target = Pose(*smoothed[:5])
            self._command(self._pose_to_motor(self.target, float(smoothed[5])), now, self._speed_scale(now))
        elif self.homing:
            self.target = self.mapper.home_pose()
            grip = self.cmd.get(GRIPPER, 30.0)
            self._command(self._pose_to_motor(self.target, grip), now, self._speed_scale(now))

    def go_to_park(self) -> None:
        """Slowly return to the pose the arm was in at startup."""
        if not self.park or self.args.dry_run:
            return
        print("Returning to start pose...")
        now = time.monotonic()
        self._sync_to_robot(now)
        t_end = now + 6.0
        while time.monotonic() < t_end:
            now = time.monotonic()
            self._command(self.park, now, scale=0.3)
            if max(abs(self.cmd[k] - self.park[k]) for k in self.park) < 0.5:
                break
            time.sleep(1 / 50)
        time.sleep(0.5)

    def compose(self, frame, obs, fps: float) -> np.ndarray:
        if obs is not None:
            draw_hand(frame, obs, (0, 255, 120) if self.tracking else (160, 160, 160))
        h, w = frame.shape[:2]
        q = self.jmap.to_model(self.cmd) if self.cmd else JointAngles()
        panel = draw_arm_panel(self.cfg.geometry, q, self.target, self.reachable, size=(int(h * 0.75), h))

        color = (80, 220, 80) if self.tracking else (0, 200, 255)
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, self.status, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        info = f"{fps:4.1f} fps"
        if obs is not None:
            info += f" | pinch {obs.pinch:.2f}"
        if self.cmd:
            info += f" | grip {self.cmd.get(GRIPPER, 0):.0f}"
        if not self.reachable and self.tracking:
            info += " | OUT OF REACH"
        flags = f"pitch:{'on' if self.cfg.mapping.track_pitch else 'off'} roll:{'on' if self.cfg.mapping.track_roll else 'off'}"
        cv2.putText(frame, info, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "SPACE track  C centre  H home  P/R pitch/roll  Q quit   " + flags,
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        return np.hstack([frame, panel])

    def run(self) -> None:
        a = self.args
        source = int(a.camera) if a.camera.isdigit() else a.camera
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera/video '{a.camera}'")

        self.arm.connect()
        writer = None
        try:
            t0 = time.monotonic()
            self._sync_to_robot(t0)
            self.park = {k: v for k, v in self.cmd.items()}
            self.arm.write(self.cmd)  # hold current position

            fps, last, n = 0.0, t0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # selfie view (also what MediaPipe expects for handedness)
                now = time.monotonic()
                obs = self.tracker.process(frame, int((now - t0) * 1000))

                if a.auto_engage and not self.tracking and not self.homing and obs is not None:
                    self.start_tracking(obs, now)
                self.step(obs, now)

                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3) if n else 0.0
                last, n = now, n + 1
                view = self.compose(frame, obs, fps)

                if a.record is not None:
                    if writer is None:
                        writer = cv2.VideoWriter(str(a.record), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                 (view.shape[1], view.shape[0]))
                    writer.write(view)
                if not a.headless:
                    cv2.imshow("SO-101 hand mirror", view)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self.handle_key(key, obs, now):
                        break
                if a.max_frames and n >= a.max_frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.go_to_park()
            finally:
                self.arm.disconnect()
                self.tracker.close()
                cap.release()
                if writer is not None:
                    writer.release()
                if not a.headless:
                    cv2.destroyAllWindows()


def main(argv=None):
    HandMirror(parse_args(argv)).run()


if __name__ == "__main__":
    main()
