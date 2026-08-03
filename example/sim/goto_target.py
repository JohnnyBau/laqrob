#!/usr/bin/env python3
"""Faehrt in MeshCat (ohne Hardware) eine feste Zielposition an.

Orientierung bleibt die der Nullstellung (Roll/Pitch/Yaw des Endeffektors bei
q=neutral) -- es wird nur x/y/z veraendert.

Nutzung:
    uv run python example/sim/goto_target.py
    uv run python example/sim/goto_target.py --x 0.2 --y 0.0 --z 0.7
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id
from example.sim.visualizer import Visualizer
from example.sim.poses_sim import move_to_pose

DEFAULT_DURATION = 3.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.2)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.7)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    args = parser.parse_args()

    print("Lade MeshCat-Visualisierer...")
    viz = Visualizer()
    model = viz.model
    end_frame_id = get_end_effector_frame_id(model)

    q_start = pin.neutral(model).copy()
    viz.update(q_start)

    # Orientierung der Nullstellung beibehalten, nur Position aendern.
    _, _, T_neutral = compute_fk(model, q_start)
    roll, pitch, yaw = pin.rpy.matrixToRpy(T_neutral[:3, :3])

    pose = {
        "name": "target", "x": args.x, "y": args.y, "z": args.z,
        "roll": roll, "pitch": pitch, "yaw": yaw,
    }
    print(f"Fahre Ziel an: x={args.x:.3f} y={args.y:.3f} z={args.z:.3f}")
    move_to_pose(viz, model, end_frame_id, q_start, pose, args.duration)
    print("Ziel erreicht (falls IK konvergiert ist, siehe ggf. [WARN] oben).")

    time.sleep(2.0)


if __name__ == "__main__":
    main()
