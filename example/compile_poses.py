#!/usr/bin/env python3
"""Berechnet aus den editierbaren x/y/z/roll/pitch/yaw-Werten in poses.json
zuverlaessig die Gelenkwinkel 'q' neu (IK, kein Hardwarezugriff noetig).

Workflow:
    1. poses.json von Hand bearbeiten (x/y/z/roll/pitch/yaw/duration/gripper).
    2. uv run python example/compile_poses.py
    3. example/playback.py fuehrt danach ueber move_to_q_traj() aus (robust,
       keine IK zur Laufzeit, keine Sprung-Gefahr).

Zuverlaessigkeit: die IK wird pro Pose mit dem VORHERIGEN q als Seed geloest
(erst die eigene alte q, sonst die der vorherigen Pose in der Liste) --
dadurch bleibt die IK i.d.R. in derselben Armkonfiguration, statt bei
kleinen Aenderungen zu einer voellig anderen (z.B. Ellbogen gespiegelten)
Loesung zu springen (siehe move_to_traj()-Sicherheitsabbruch/CLIK-Vorfall).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reBotArm_control_py.kinematics import (
    compute_fk, pos_rot_to_se3, get_end_effector_frame_id, load_robot_model, pad_q_for_model,
)
from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik_with_retry, IKParams

from _common import load_poses, save_poses, POSES_FILE

IK_PARAMS = IKParams(max_iter=200, tolerance=2e-3, step_size=0.5, damping=1e-6)
IK_MAX_RETRIES = 150


def main() -> None:
    model = load_robot_model()
    end_frame_id = get_end_effector_frame_id(model)
    data = model.createData()
    n = model.nq

    poses = load_poses(POSES_FILE)
    if not poses:
        print(f"Keine Posen in {POSES_FILE} gefunden.")
        return

    q_prev = np.zeros(n)
    n_fail = 0

    for pose in poses:
        # eigenes altes q bevorzugen (stabiler Seed), sonst das der Vorpose.
        seed = np.array(pose["q"]) if "q" in pose else q_prev
        seed = pad_q_for_model(model, seed, n)

        T_target = pos_rot_to_se3(
            np.array([pose["x"], pose["y"], pose["z"]]),
            roll=pose["roll"], pitch=pose["pitch"], yaw=pose["yaw"],
        )
        result = solve_ik_with_retry(
            model, data, end_frame_id, T_target, seed.copy(), IK_PARAMS,
            max_retries=IK_MAX_RETRIES,
        )
        if not result.success:
            print(f"-> {pose['name']}: IK FEHLGESCHLAGEN err={result.error:.4f} -- q bleibt unveraendert")
            n_fail += 1
            q_prev = seed
            continue

        q_new = pad_q_for_model(model, result.q, n)
        delta = float(np.abs(q_new - seed).max()) if "q" in pose else 0.0
        flag = " (q neu, kein Vergleich)" if "q" not in pose else f" (max. Aenderung {delta:.4f} rad)"
        print(f"-> {pose['name']}: OK{flag}")
        pose["q"] = [float(v) for v in q_new[:n]]
        q_prev = q_new

    save_poses(poses, POSES_FILE)
    ok = len(poses) - n_fail
    print(f"\n{ok}/{len(poses)} Posen erfolgreich kompiliert, in {POSES_FILE} gespeichert.")


if __name__ == "__main__":
    main()
