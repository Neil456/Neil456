# SO-101 Hand Mirror 🖐️🦾

Move your hand in front of a webcam and an SO-101 arm copies it in real time.
No gloves, no markers, no leader arm. Just MediaPipe hand tracking plus
closed-form inverse kinematics.

```
webcam ─► MediaPipe (21 hand landmarks)
       ─► hand signals: palm position, distance, finger pitch, roll, pinch
       ─► gripper target pose (x, y, z, pitch, roll)      mapping.py
       ─► One Euro smoothing                              filters.py
       ─► analytic IK ─► joint limits                     kinematics.py
       ─► model angles ─► motor degrees ─► speed limit    robot.py
       ─► LeRobot send_action ─► SO-101
```

| Your hand                         | The arm                         |
|-----------------------------------|---------------------------------|
| move left / right / up / down     | gripper moves the same way, like a mirror |
| move toward / away from the camera| gripper reaches toward / away from you |
| tilt fingers down / up            | gripper pitches down / up       |
| rotate hand                       | wrist roll                      |
| pinch thumb + index               | close gripper                   |

Setup: put the arm next to the webcam, **facing you**.

---

## 1. Install

LeRobot needs Python ≥ 3.12.

```bash
cd so101-hand-mirror
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "lerobot[feetech]"     # robot driver (skip if you only want --dry-run)
```

The MediaPipe hand model (~8 MB) downloads automatically on first run into `models/`.

## 2. Try it without the robot

```bash
python -m hand_mirror.run --dry-run
```

You'll see the webcam with your hand skeleton on the left, and a simulated arm
(side and top view) on the right. Press **space** with your hand in view to
start tracking. Get a feel for the controls here before involving hardware.

## 3. LeRobot calibration (if you haven't already)

If you followed the SO-101 assembly guide you've probably already done this.
The `--robot.id` you pick here is the `--robot-id` you'll use below.

```bash
lerobot-find-port
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
```

## 4. Pose calibration (once, about 2 minutes)

LeRobot's calibration tells us the *range* of each joint. IK also needs to know
which motor reading corresponds to which real-world angle, and which way each
servo turns. This script finds that out. **Torque goes off, so hold the arm.**

```bash
python -m hand_mirror.calibrate_pose --port /dev/ttyACM0 --robot-id my_awesome_follower_arm
```

1. Put the arm in the **L pose**: base centred, upper arm straight up, forearm
   horizontal pointing forward, gripper in line with the forearm and not rolled.
   Press Enter.
2. For each joint, move it about 30° in the direction it asks and press Enter.

Writes `calibration.json`. Then verify it:

```bash
python -m hand_mirror.calibrate_pose --port /dev/ttyACM0 --robot-id my_awesome_follower_arm --check
```

Move the arm by hand and check that the printed tip position makes sense (use
a ruler: `z` is the gripper tip's height above the table in cm). If it's off by
a lot, re-check the L pose, or measure your link lengths and put them in
`config.yaml → geometry`.

## 5. Run it

```bash
python -m hand_mirror.run --port /dev/ttyACM0 --robot-id my_awesome_follower_arm
```

| Key     | Action |
|---------|--------|
| `space` | start / pause tracking. Your current hand pose becomes "neutral", which maps to the arm's home pose |
| `c`     | re-centre (current hand pose becomes neutral) |
| `h`     | stop tracking and move to the home pose |
| `p`/`r` | toggle wrist pitch / roll tracking |
| `q`/Esc | quit. The arm slowly returns to where it started, then torque is released |

Tips: good lighting, a plain background, and the hand 40–70 cm from the
camera. Start with `p` and `r` off (pitch and roll) until the position
tracking feels right.

### Safety features
- Nothing moves until you press **space**. For the first 2 s after that, the
  arm moves at 20–100% speed so it eases into position.
- Per-joint speed limit (`control.max_speed_deg_s`) plus LeRobot's per-step
  clamp (`max_relative_target`).
- The gripper tip is kept inside a box (`mapping.x/y/z_range`), so it can't be
  commanded into the table. Software joint limits apply on top, and so do the
  servo limits from LeRobot calibration.
- If your hand disappears for 0.6 s, tracking stops and the arm holds its position.
- **Keep a hand near the power switch the first few times.**

---

## Tuning: "it moves the wrong way"

Everything is in `config.yaml`.

| Symptom | Fix |
|---|---|
| Arm goes left when you go right | `mapping.lateral_sign: -1` (or you placed the robot facing away from you) |
| Up/down or toward/away reversed | `vertical_sign` / `depth_sign: -1` |
| Wrist pitch or roll reversed | `pitch_sign` / `roll_sign: -1` |
| One joint goes the opposite way from the visualization | re-run `calibrate_pose` and check the direction for that joint |
| Gripper open/close reversed | swap `gripper_closed` and `gripper_open` |
| Gripper never fully closes/opens | watch `pinch` in the HUD and adjust `pinch_closed` / `pinch_open` |
| Too jittery | lower `smoothing.min_cutoff` (e.g. 0.6) |
| Too laggy | raise `smoothing.beta` or `min_cutoff` |
| Moves too little / too much | `lateral_gain`, `vertical_gain`, `depth_gain` |
| Gets close to the table | raise `z_range[0]` |

---

## How the IK works

Joints 2–4 of the SO-101 (shoulder lift, elbow, wrist flex) rotate about
parallel horizontal axes. So:

1. **Base:** `pan = atan2(y, x)` turns the arm to face the target. Everything
   left is a 2D problem in the vertical plane: `r` = horizontal distance, `z` = height.
2. **Wrist centre:** step back from the target along the desired gripper pitch φ:
   `w = tip − L3·(cos φ, sin φ)`.
3. **Shoulder + elbow:** a 2-link arm reaching `w`, solved with the law of cosines:
   `cos(elbow) = (d² − L1² − L2²) / (2·L1·L2)` (we take the elbow-up solution), then
   `shoulder = atan2(wz, wr) − atan2(L2·sin(elbow), L1 + L2·cos(elbow))`.
4. **Wrist pitch:** whatever is left: `wrist = φ − shoulder − elbow`.
5. **Roll:** directly from your hand's roll.

If a target is out of reach, the wrist centre is pulled to the edge of the
workspace and the HUD shows `OUT OF REACH`. It's exact and runs in microseconds,
with no iterative solver needed.

Depth from a single webcam uses the pinhole model: `distance ∝ real size / image size`.
MediaPipe's *world* landmarks give the palm's real 3D shape, so tilting your
hand doesn't get mistaken for moving it closer.

## Project layout

```
hand_mirror/
  run.py             main loop, keyboard, HUD
  hand_tracker.py    MediaPipe wrapper -> palm position, depth, pitch, roll, pinch
  mapping.py         hand signals -> gripper target pose (relative to "neutral")
  kinematics.py      forward + inverse kinematics, joint limits
  filters.py         One Euro filter, joint speed limiter
  robot.py           LeRobot SO-101 wrapper, motor<->model angle mapping, dry-run arm
  calibrate_pose.py  L-pose + direction calibration, live --check readout
  viz.py             side/top view of the arm
config.yaml          all the knobs
tests/               IK round-trip, mapping, filters, MediaPipe on a real photo
```

Run the tests with `python -m pytest`.

## Ideas for next steps
- Use two hands: one drives position, the other's gestures switch modes.
- Record demonstrations with this and train a LeRobot policy on them.
- Swap the analytic IK for the URDF + `placo` solver LeRobot ships, to get full
  5-DOF orientation control.
- Add a second camera for true stereo depth.
