#!/usr/bin/env python3
"""Spielt eine Gelenk-Posen-JSON am echten Arm ab (MIT + Gravitations-Feedforward).

Geschwindigkeit:
    --speed   maximale Gelenkgeschwindigkeit [rad/s] fuer Posen ohne 'duration'-Feld
    --scale   Zeitdehnungsfaktor (>1 = langsamer/sicherer); wirkt auf ALLE Posen

Wenn eine Pose kein 'duration'-Feld hat, wird die Dauer aus dem groessten
Gelenkwinkel-Delta geteilt durch --speed berechnet (mind. MIN_DURATION Sekunden),
dann mit --scale multipliziert.

Nutzung:
    uv run python example/play.py --poses example/poses_body1_playback.json --scale 5.0
    uv run python example/play.py --poses example/poses_body1_playback.json --scale 1.0 --speed 1.0

Sicherheit: bricht bei Fehler sofort ab (haelt aktuelle Pose); fuehrt am Ende
immer ctrl.end() aus (Safe-Home + Disconnect).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose

from _common import GRIPPER_TAU_OPEN, GRIPPER_TAU_CLOSE, gripper_wait_for_event

MIN_DURATION = 0.5  # Mindest-Bewegungszeit auch bei sehr kleinen Deltas [s]


def load_poses(path: Path) -> list:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data["poses"]


def run(poses_path: Path, scale: float, speed: float, dt: float) -> None:
    poses = load_poses(poses_path)
    if not poses:
        print(f"Keine Posen in {poses_path}.")
        return

    rebotarm = RebotArm()
    ctrl = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True, dt=dt)
    ctrl.start()

    total = len(poses)
    aborted = False
    try:
        for idx, pose in enumerate(poses, start=1):
            name = pose.get("name", f"#{idx}")
            if "q" not in pose:
                print(f"[{idx}/{total}] {name}  FEHLER: kein 'q' in Pose — JSON mit poses_traj_sim.py --save-q kompilieren!")
                aborted = True
                break
            q_target = np.array(pose["q"], dtype=float)

            pose_speed = pose.get("speed", speed)
            if "duration" in pose:
                duration = pose["duration"] * scale
            else:
                # Dauer aus aktuellem Ist-Zustand berechnen
                q_now, _, _ = rebotarm.get_state()
                dist = float(np.max(np.abs(q_target - q_now[: len(q_target)])))
                duration = max(MIN_DURATION, dist / pose_speed) * scale

            print(f"[{idx}/{total}] {name}  speed={pose_speed} rad/s  duration={duration:.2f}s")
            ok = ctrl.move_to_q_traj(q_target, duration=duration)
            if not ok:
                print(f"  ABBRUCH: Fehler bei '{name}'. Sequenz gestoppt.")
                aborted = True
                break

            while ctrl._moving:
                time.sleep(dt)

            gripper = pose.get("gripper", "none")
            if gripper == "open":
                ctrl.open_gripper(tau=GRIPPER_TAU_OPEN)
                event, g_pos, travel = gripper_wait_for_event(rebotarm, send=False)
                print(f"  Greifer offen: {event}  pos={g_pos:+.4f} rad  travel={travel:.3f} rad")
            elif gripper == "close":
                ctrl.close_gripper(tau=GRIPPER_TAU_CLOSE)
                event, g_pos, travel = gripper_wait_for_event(rebotarm, send=False)
                print(f"  Greifer zu:    {event}  pos={g_pos:+.4f} rad  travel={travel:.3f} rad")

            time.sleep(0.2)
    finally:
        ctrl.end()
        print("Abgebrochen." if aborted else "Fertig.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=Path, required=True, help="Pfad zur Gelenk-Posen-JSON")
    parser.add_argument(
        "--scale", type=float, default=3.0,
        help="Zeitdehnungsfaktor (>1 = langsamer/sicherer, 1.0 = Normaltempo). Standard: 3.0",
    )
    parser.add_argument(
        "--speed", type=float, default=0.5,
        help="Max. Gelenkgeschwindigkeit [rad/s] fuer Posen ohne 'duration'-Feld. Standard: 0.5",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02,
        help="Trajektorien-Zeitschritt [s] (50 Hz). Standard: 0.02",
    )
    args = parser.parse_args()
    run(args.poses, args.scale, args.speed, args.dt)


if __name__ == "__main__":
    main()
