#!/usr/bin/env python3
"""Konvertiert eine Kartesisch-Posen-JSON offline zu Gelenkwinkeln (kein MeshCat).

Fuer jede Pose wird IK geloest; der Seed ist das Ergebnis der vorherigen Pose
(stabile Armkonfiguration ohne Sprung).

Nutzung:
    uv run python example/sim/compile_q.py example/poses_body1_pick_paint_drop.json
    uv run python example/sim/compile_q.py example/poses_body1_pick_paint_drop.json \
        --out example/poses_body1_pick_paint_drop_q.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vec_to_rpy import vec_to_rpy
from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik_with_retry, IKParams
from example.sim.traj_sim import make_pose

IK_PARAMS = IKParams(max_iter=300, tolerance=1e-4, step_size=0.8, damping=1e-6)
IK_MAX_RETRIES = 150


def load_poses(path: Path) -> list:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["poses"]


def with_rpy(p: dict) -> dict:
    if "roll" not in p and "vx" in p:
        roll_rad = math.radians(p.get("roll_deg", 0.0))
        x, y, z, roll, pitch, yaw = vec_to_rpy(
            p["x"], p["y"], p["z"], p["vx"], p["vy"], p["vz"], roll_rad
        )
        return {**p, "roll": roll, "pitch": pitch, "yaw": yaw}
    return p


def compile_poses(poses: list, model, end_frame_id, data) -> list:
    q_last = pin.neutral(model).copy()
    n = model.nq
    compiled = []
    n_fail = 0

    total = len(poses)
    for idx, pose in enumerate(poses, start=1):
        name = pose.get("name", f"#{idx}")

        if "j6_sweep" in pose and "x" not in pose and "q" not in pose:
            q_arm = q_last.copy()
            q_arm[5] = pose["j6_sweep"]
            q_last = q_arm
            compiled.append({**pose, "q": [float(v) for v in q_last[:6]]})
            print(f"  [{idx}/{total}] {name}  j6={pose['j6_sweep']:.3f} rad")

        elif "x" not in pose and "q" in pose:
            q_last = np.array(pose["q"], dtype=float)
            if len(q_last) < n:
                q_last = np.pad(q_last, (0, n - len(q_last)))
            compiled.append({**pose, "q": [float(v) for v in q_last[:6]]})
            print(f"  [{idx}/{total}] {name}  (Gelenkraum)")

        else:
            pose = with_rpy(pose)
            j6_seed = pose.get("j6")
            target = make_pose(pose["x"], pose["y"], pose["z"],
                               pose["roll"], pose["pitch"], pose["yaw"])
            seed = q_last.copy()
            if j6_seed is not None:
                seed[5] = j6_seed

            result = solve_ik_with_retry(
                model, data, end_frame_id, target, seed, IK_PARAMS,
                max_retries=IK_MAX_RETRIES,
            )

            if not result.success:
                print(f"  [{idx}/{total}] {name}  IK FEHLGESCHLAGEN (err={result.error:.4f})")
                compiled.append({**pose, "ik_failed": True})
                n_fail += 1
                continue

            q_res = result.q.copy()
            if j6_seed is not None:
                q_res[5] = j6_seed
            q_last = q_res
            compiled.append({**pose, "q": [float(v) for v in q_last[:6]]})
            print(f"  [{idx}/{total}] {name}  OK  q=[{', '.join(f'{v:.3f}' for v in q_last[:6])}]")

    return compiled, n_fail


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("poses_file", type=Path, help="Eingabe-JSON (kartesisch)")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Ausgabedatei (Standard: <eingabe>_q.json)",
    )
    args = parser.parse_args()

    out_path = args.out or args.poses_file.with_stem(args.poses_file.stem + "_q")

    model = load_robot_model()
    end_frame_id = get_end_effector_frame_id(model)
    data = model.createData()

    poses = load_poses(args.poses_file)
    print(f"{len(poses)} Posen geladen aus {args.poses_file}\n")

    compiled, n_fail = compile_poses(poses, model, end_frame_id, data)

    out_path.write_text(json.dumps(compiled, indent=2, ensure_ascii=False))
    ok = len(compiled) - n_fail
    print(f"\n{ok}/{len(compiled)} OK, {n_fail} Fehler  →  {out_path}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
