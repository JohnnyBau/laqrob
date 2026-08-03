#!/usr/bin/env python3
"""Posen-Wiedergabe-Simulation — spielt poses.json in MeshCat ab (ohne Hardware).

Laedt eine Liste von Posen aus einer JSON-Datei und faehrt sie nacheinander in
der MeshCat-Visualisierung an (SE(3)-Geodaete + CLIK-Tracking, gleicher Ansatz
wie traj_sim.py).

Jede Pose braucht x, y, z (Meter) und GENAU EINE der beiden Orientierungsarten:

    - roll, pitch, yaw (Radiant)         -- klassische Euler-Winkel
    - direction [x, y, z] (+ optional up [x, y, z], Default [0,0,1])
      -- Werkzeugrichtungsvektor: die Werkzeugspitze (lokale X-Achse des
         Endeffektors) zeigt entlang `direction`. `up` legt nur die Rotation
         um die eigene Achse fest (Roll) und muss nicht exakt senkrecht sein.

Nutzung:
    uv run python example/sim/poses_sim.py [pfad/zu/poses.json]

Standard-Datei: example/sim/poses.json
Beenden: Strg+C
"""

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id, pad_q_for_model
from reBotArm_control_py.kinematics.inverse_kinematics import (
    pos_rot_to_se3,
    solve_ik_with_retry,
    IKParams as IKSolverParams,
)
from reBotArm_control_py.trajectory import (
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
    TrajProfile,
    TrajPlanParams,
    IKParams,
)
from example.sim.visualizer import Visualizer

DEFAULT_POSES_FILE = Path(__file__).resolve().parent / "poses.json"
DEFAULT_DURATION = 2.5
PAUSE_BETWEEN_POSES = 0.5

should_exit = False


def signal_handler(sig, frame):
    global should_exit
    should_exit = True


def load_poses(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def direction_to_rot(direction, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Baut eine Rotationsmatrix, deren lokale X-Achse (Werkzeugspitze) entlang `direction` zeigt.

    `up` bestimmt nur die Rotation um die Werkzeugachse (Roll) und muss nicht
    exakt senkrecht zu `direction` stehen.
    """
    x = np.asarray(direction, dtype=float)
    x /= np.linalg.norm(x)
    up_vec = np.asarray(up, dtype=float)
    if abs(np.dot(x, up_vec / np.linalg.norm(up_vec))) > 0.99:
        up_vec = np.array([0.0, 1.0, 0.0])  # Fallback, falls up ~parallel zu direction
    y = np.cross(up_vec, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def pose_target_rot(pose: dict) -> np.ndarray:
    """Liefert die Ziel-Rotationsmatrix -- entweder aus `direction`(+`up`) oder roll/pitch/yaw."""
    if "direction" in pose:
        return direction_to_rot(pose["direction"], pose.get("up", (0.0, 0.0, 1.0)))
    return pin.rpy.rpyToMatrix(
        pose.get("roll", 0.0), pose.get("pitch", 0.0), pose.get("yaw", 0.0)
    )


def move_to_pose(viz, model, end_frame_id, q_start, pose, duration, dt=1.0 / 50.0):
    """Plant + animiert eine SE(3)-Geodaete von q_start zur Zielpose, gibt q am Ziel zurueck."""
    data = model.createData()
    ik_params = IKSolverParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

    target = pos_rot_to_se3(
        np.array([pose["x"], pose["y"], pose["z"]]),
        rot=pose_target_rot(pose),
    )
    # Mit den in der Pose gespeicherten Gelenkwinkeln seeden statt mit der vorherigen
    # Armstellung -- vermeidet, dass die IK in einen anderen (falschen) Loesungsast springt.
    if "q" in pose:
        q_seed = pad_q_for_model(model, np.array(pose["q"], dtype=float))
    else:
        q_seed = q_start.copy()
    ik_result = solve_ik_with_retry(model, data, end_frame_id, target, q_seed, ik_params)
    if not ik_result.success:
        print(f"  [WARN] IK fuer Pose '{pose.get('name', '?')}' nicht konvergiert "
              f"(err={ik_result.error:.3e}) -- ueberspringe")
        return q_start

    q_end = ik_result.q
    T_start = compute_fk(model, q_start)[2]
    T_end = compute_fk(model, q_end)[2]

    traj_params = TrajPlanParams(dt=dt, profile=TrajProfile.MIN_JERK)
    clik_params = IKParams(max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8)

    cart_result = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, traj_params)
    joint_traj = track_trajectory(
        model, end_frame_id, cart_result.trajectory, q_start, clik_params, null_gain=0.1
    )

    ref_positions = [
        pt.pose.translation.tolist() if hasattr(pt.pose, "translation") else pt.pose[:3, 3].tolist()
        for pt in cart_result.trajectory.points()
    ]
    viz.clear_paths()
    viz.draw_ref_path(ref_positions)

    times = [pt.time for pt in joint_traj]
    visited = []
    for i, pt in enumerate(joint_traj):
        if should_exit:
            break
        viz.update(pt.q)
        _, _, T = compute_fk(model, pt.q)
        visited.append(T[:3, 3].tolist())
        viz.draw_actual_path(visited)
        if i < len(times) - 1:
            time.sleep(max(0.002, times[i + 1] - times[i]))

    return q_end


def main():
    signal.signal(signal.SIGINT, signal_handler)

    args = [a for a in sys.argv[1:] if a != "--loop"]
    loop = "--loop" in sys.argv[1:]  # endlos zwischen den Posen hin- und herfahren, Strg+C zum Beenden
    poses_path = Path(args[0]) if args else DEFAULT_POSES_FILE
    poses = load_poses(poses_path)
    print(f"{len(poses)} Pose(n) geladen aus {poses_path}")

    print("Lade MeshCat-Visualisierer...")
    viz = Visualizer()
    model = viz.model
    end_frame_id = get_end_effector_frame_id(model)

    q = pin.neutral(model).copy()
    viz.update(q)
    print("Roboter in Nullstellung. Wiedergabe startet...\n")

    durchlauf = 0
    while not should_exit:
        durchlauf += 1
        for idx, pose in enumerate(poses):
            if should_exit:
                break
            name = pose.get("name", f"pose_{idx}")
            duration = pose.get("duration", DEFAULT_DURATION)
            prefix = f"(Durchlauf {durchlauf}) " if loop else ""
            print(f"{prefix}[{idx + 1}/{len(poses)}] -> '{name}'  "
                  f"pos=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f})")
            q = move_to_pose(viz, model, end_frame_id, q, pose, duration)
            time.sleep(PAUSE_BETWEEN_POSES)
        if not loop:
            break

    print("\nWiedergabe beendet." if not should_exit else "\nAbgebrochen.")
    print("Fenster bleibt offen, damit du dir die Endpose ansehen kannst -- Strg+C zum Beenden.")
    while not should_exit:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
