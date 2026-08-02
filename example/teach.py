#!/usr/bin/env python3
"""Teach-Modus: Arm per Free-Drive (Gravitationskompensation) von Hand fuehren,
Posen inkl. Greifer-Aktion in poses.json speichern.

Nutzung:
    uv run python example/teach.py
    Enter / <Name> + Enter  -> aktuelle Pose speichern
    q + Enter               -> beenden
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.kinematics import joint_to_pose

from _common import gravity_hold, load_poses, save_poses, POSES_FILE


def main() -> None:
    rebotarm = RebotArm()
    rebotarm.connect()
    rebotarm.arm.mode_mit()
    if rebotarm.has_gripper:
        rebotarm.gripper.mode_mit()
    rebotarm.enable_all()
    rebotarm.start_control_loop(gravity_hold, rate=rebotarm.rate)

    print("=" * 60)
    print("  Free-Drive aktiv -- Arm haelt Pose, laesst sich von Hand fuehren.")
    print("  Enter / <Name> + Enter  -> aktuelle Pose speichern")
    print("  q + Enter               -> beenden und poses.json schreiben")
    print("=" * 60)

    poses = load_poses()
    try:
        while True:
            line = input("> ").strip()
            if line.lower() == "q":
                break

            q, _, _ = rebotarm.get_state()
            pos, rpy = joint_to_pose(q)
            n_arm = rebotarm.arm.num_joints

            gripper_in = input("  Greifer [o=open / c=close / Enter=keine Aktion]: ").strip().lower()
            gripper_action = {"o": "open", "c": "close"}.get(gripper_in, "none")

            entry = {
                "name": line if line else f"pose_{len(poses) + 1}",
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2]),
                "duration": 2.0,
                "gripper": gripper_action,
                # reale Gelenkwinkel: playback.py fährt darauf direkt (Min-Jerk,
                # keine IK) statt die Pose neu zu loesen -- vermeidet CLIK-
                # Mehrdeutigkeiten/Sprünge zwischen weit auseinanderliegenden Posen.
                "q": [float(v) for v in q[:n_arm]],
            }
            poses.append(entry)
            save_poses(poses)  # inkrementell schreiben, kein Datenverlust bei Absturz
            print(f"  gespeichert: {entry['name']}  "
                  f"pos=({entry['x']:+.3f},{entry['y']:+.3f},{entry['z']:+.3f})  "
                  f"gripper={gripper_action}")
    finally:
        print(f"\n{len(poses)} Posen in {POSES_FILE} gespeichert.")
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
