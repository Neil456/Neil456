"""Teach the code how your motors map to the kinematic model.

    python -m hand_mirror.calibrate_pose --port /dev/ttyACM0 --robot-id my_follower
    python -m hand_mirror.calibrate_pose --port /dev/ttyACM0 --robot-id my_follower --check

Torque is switched OFF so you can move the arm by hand. Hold it - it will go limp!

Step 1 - put the arm in the reference "L pose" and press Enter:

          gripper    forearm
            ==<|=============o  <- elbow
                             |
                             |  upper arm (vertical)
                             |
                           [base]   <- arm pointing straight forward (pan centred)

   * upper arm straight up (line from shoulder axis to elbow axis vertical)
   * forearm horizontal, pointing forward
   * gripper in line with the forearm, jaws level (not rolled)

Step 2 - for each joint, move it ~30 deg in the direction asked and press Enter.
That tells us which way each servo counts.

--check prints live joint angles + gripper tip position while you move the arm
by hand, so you can sanity-check the result (e.g. measure tip height with a ruler).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import load_config
from .kinematics import forward
from .robot import MOTOR_NAMES, REFERENCE_POSE, JointCal, JointMap, SO101Arm
from .run import DEFAULT_CAL_PATH

DIRECTIONS = {
    "pan": "rotate the whole arm to ITS OWN LEFT (counter-clockwise seen from above)",
    "shoulder": "tilt the upper arm BACKWARD (away from the gripper side)",
    "elbow": "raise the forearm so the gripper end goes UP",
    "wrist_pitch": "tilt the gripper tip UP",
    "wrist_roll": "roll the gripper CLOCKWISE, looking from the wrist toward the gripper tip",
}


def calibrate(arm: SO101Arm, out: Path) -> None:
    arm.torque(False)
    print(__doc__.split("Step 1")[1].split("--check")[0])
    input(">>> Put the arm in the L pose, then press Enter...")
    ref = arm.read()
    print("Recorded:", {MOTOR_NAMES[j]: round(ref[MOTOR_NAMES[j]], 1) for j in MOTOR_NAMES})

    cal: dict[str, JointCal] = {}
    for joint, motor in MOTOR_NAMES.items():
        input(f"\n>>> [{motor}] {DIRECTIONS[joint]} ~30 deg, then press Enter...")
        delta = arm.read()[motor] - ref[motor]
        if abs(delta) < 5:
            print(f"  only moved {delta:.1f} deg - assuming +1. Re-run if that's wrong.")
            sign = 1.0
        else:
            sign = 1.0 if delta > 0 else -1.0
            print(f"  moved {delta:+.1f} motor deg -> sign {sign:+.0f}")
        cal[joint] = JointCal(
            motor_ref=ref[motor], model_ref=getattr(REFERENCE_POSE, joint), sign=sign
        )
        input("  move it back to the L pose and press Enter...")

    JointMap(cal).save(out)
    print(f"\nSaved {out}. Next: python -m hand_mirror.calibrate_pose --check ... to verify.")


def check(arm: SO101Arm, cal_path: Path, config: Path | None) -> None:
    arm.torque(False)
    jmap = JointMap.load(cal_path)
    geom = load_config(config).geometry
    print("Move the arm by hand. Ctrl+C to stop.\n")
    try:
        while True:
            q = jmap.to_model(arm.read())
            tip = forward(geom, q).tip
            print(
                f"\rpan {q.pan:6.1f} sh {q.shoulder:6.1f} elb {q.elbow:6.1f} "
                f"wp {q.wrist_pitch:6.1f} roll {q.wrist_roll:6.1f} | "
                f"tip x {tip[0]*100:5.1f} y {tip[1]*100:5.1f} z {tip[2]*100:5.1f} cm   ",
                end="",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="my_awesome_follower_arm")
    p.add_argument("--out", type=Path, default=DEFAULT_CAL_PATH)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--check", action="store_true", help="Live readout using an existing calibration")
    args = p.parse_args(argv)

    arm = SO101Arm(args.port, args.robot_id, max_relative_target=None)
    arm.connect()
    print("\n!!! Torque is now OFF - support the arm so it doesn't fall. !!!\n")
    try:
        if args.check:
            check(arm, args.out, args.config)
        else:
            calibrate(arm, args.out)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
