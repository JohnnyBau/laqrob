#!/usr/bin/env python3
"""Spielt eine Posen-JSON direkt ab (kein Subprocess).

Nutzung:
    uv run python example/sim/poses_traj_sim.py example/poses.json
    uv run python example/sim/poses_traj_sim.py example/poses.json --prefix pick
    uv run python example/sim/poses_traj_sim.py example/poses.json --names pick_pre_1 pick_rdy_1
    uv run python example/sim/poses_traj_sim.py example/poses.json --pause 3.0
"""
import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vec_to_rpy import vec_to_rpy
from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id
from reBotArm_control_py.trajectory import IKParams, TrajProfile
from example.sim.visualizer import Visualizer
import example.sim.traj_sim as _sim
from example.sim.traj_sim import run_joint_trajectory, run_trajectory, _solve_ik, make_pose

DEFAULT_PAUSE = 0
ANGULAR_SPEED = 1.5  # rad/s fuer Orientierungsaenderungen (z.B. j6-Drehung)


def load_poses(path: Path) -> list:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data["poses"]


def with_rpy(poses: list) -> list:
    """Ergaenzt roll/pitch/yaw via vec_to_rpy(), falls eine Pose nur vx/vy/vz hat."""
    result = []
    for p in poses:
        if "roll" not in p and "vx" in p:
            roll_rad = math.radians(p.get("roll_deg", 0.0))
            x, y, z, roll, pitch, yaw = vec_to_rpy(
                p["x"], p["y"], p["z"], p["vx"], p["vy"], p["vz"], roll_rad
            )
            p = {**p, "roll": roll, "pitch": pitch, "yaw": yaw}
        result.append(p)
    return result


def select_poses(poses, names=None, prefix=None):
    if names:
        by_name = {p["name"]: p for p in poses if "name" in p}
        missing = [n for n in names if n not in by_name]
        if missing:
            print(f"WARNUNG: Posen nicht gefunden: {missing}", file=sys.stderr)
        return [by_name[n] for n in names if n in by_name]
    if prefix:
        return [p for p in poses if p.get("name", "").startswith(prefix)]
    return poses


def run_sequence(poses, viz, model, end_frame_id, pause_s: float) -> list:
    """Gibt kompilierte Posen zurück (jede Pose mit 'q' ergänzt)."""
    ik_params = IKParams(max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8)
    dt = 1.0 / 50.0
    q_last = pin.neutral(model).copy()
    compiled = []

    total = len(poses)
    for idx, pose in enumerate(poses, start=1):
        if _sim.should_exit:
            break
        name = pose.get("name", f"#{idx}")
        is_j6_sweep = "j6_sweep" in pose and "x" not in pose and "q" not in pose
        is_joint_space = "x" not in pose and "q" in pose

        if is_j6_sweep:
            print(f"\n=== [{idx}/{total}] Pose '{name}' -- [j6-Sweep → {pose['j6_sweep']:.3f} rad] ===")
            q_arm = q_last[:6].copy()
            q_arm[5] = pose["j6_sweep"]
            q_last = run_joint_trajectory(viz, model, end_frame_id, q_last, q_arm, dt,
                                          speed=pose.get("speed"))
            compiled.append({**pose, "q": [float(v) for v in q_last[:6]]})

        elif is_joint_space:
            print(f"\n=== [{idx}/{total}] Pose '{name}' -- [Gelenkraum-Pose] ===")
            q_arm = np.array(pose["q"], dtype=float)
            q_last = run_joint_trajectory(viz, model, end_frame_id, q_last, q_arm, dt,
                                          speed=pose.get("speed"))
            compiled.append({**pose, "q": [float(v) for v in q_last[:6]]})

        else:
            print(f"\n=== [{idx}/{total}] Pose '{name}' -- "
                  f"x={pose['x']:.4f} y={pose['y']:.4f} z={pose['z']:.4f} ===")
            j6_seed = pose.get("j6")
            target_pose = make_pose(pose["x"], pose["y"], pose["z"],
                                    pose["roll"], pose["pitch"], pose["yaw"])
            T0 = compute_fk(model, q_last)[2]
            ik_seed = q_last.copy()
            if j6_seed is not None:
                ik_seed[5] = j6_seed
            ik_res_q, ik_success = _solve_ik(model, end_frame_id, target_pose, ik_seed, ik_params)
            if j6_seed is not None:
                ik_res_q = ik_res_q.copy()
                ik_res_q[5] = j6_seed
            if not ik_success:
                print("  IK fehlgeschlagen\n")
                compiled.append({**pose, "ik_failed": True})
                continue
            # log6 gegen FK(q_end) damit j6-Zwang in der Dauer steckt, nicht roll=0-Näherung
            T_end = compute_fk(model, ik_res_q)[2]
            log6 = pin.log6(pin.SE3(T0[:3, :3], T0[:3, 3]).inverse() *
                            pin.SE3(T_end[:3, :3], T_end[:3, 3]))
            angular_speed = pose.get("speed", ANGULAR_SPEED)
            duration = max(0.3,
                           np.linalg.norm(log6.linear) / _sim.LINEAR_SPEED,
                           np.linalg.norm(log6.angular) / angular_speed)
            _, joint_traj, _, _ = run_trajectory(
                viz=viz, model=model, end_frame_id=end_frame_id,
                q_start=q_last, q_end=ik_res_q, duration=duration,
                dt=dt, profile=TrajProfile.MIN_JERK, accel_ratio=0.25, null_gain=0.1,
            )
            q_last = joint_traj[-1].q.copy()
            compiled.append({**pose, "q": [float(v) for v in q_last[:6]]})

        viz.update(q_last)
        if pause_s > 0:
            time.sleep(pause_s)

    print("\nAlle Posen abgespielt.")
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("poses_file", type=Path, help="Pfad zur poses.json")
    parser.add_argument("--names", nargs="+", help="Nur diese Posen abspielen")
    parser.add_argument("--prefix", help="Nur Posen mit diesem Namens-Praefix")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE, help="Pause zwischen Posen [s]")
    args = parser.parse_args()

    def _on_sigint(sig, frame):
        _sim.should_exit = True
    signal.signal(signal.SIGINT, _on_sigint)

    poses = with_rpy(load_poses(args.poses_file))
    selected = select_poses(poses, args.names, args.prefix)
    if not selected:
        print("Keine passenden Posen gefunden.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(selected)} Posen ausgewaehlt aus {args.poses_file}.")

    print("Lade MeshCat...")
    viz = Visualizer(open_browser=True)
    model = viz.model
    end_frame_id = get_end_effector_frame_id(model)
    print(f"Modell: {model.nq} Gelenke\n")
    viz.update(pin.neutral(model))

    compiled = run_sequence(selected, viz, model, end_frame_id, args.pause)

    while not _sim.should_exit:
        time.sleep(0.5)


if __name__ == "__main__":
    main()

