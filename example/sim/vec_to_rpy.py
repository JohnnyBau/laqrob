#!/usr/bin/env python3
"""Wandelt x y z vx vy vz (Position + Greifer-Richtungsvektor) in x y z roll pitch yaw um.

Der Richtungsvektor (vx, vy, vz) entspricht der lokalen X-Achse des Endeffektors
(siehe live_pose_plot.py / kinematics-model-facts). Da eine reine Richtung den
Rollwinkel um die eigene Achse nicht festlegt, wird roll=0 angenommen -- exakt
dieselbe Konvention wie in direction_roll_to_matrix()/direction_to_rot()
(vec_pose_sim.py, poses_sim.py) an anderer Stelle im Projekt.

Nutzung:
    uv run python example/sim/vec_to_rpy.py x y z vx vy vz
    uv run python example/sim/vec_to_rpy.py            # interaktiv, eine Zeile pro Pose
"""
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics.inverse_kinematics import direction_roll_to_matrix


def vec_to_rpy(x, y, z, vx, vy, vz, roll_rad=0.0):
    """(x,y,z,vx,vy,vz) -> (x,y,z,roll,pitch,yaw); roll_rad dreht um die Annaeherungsachse."""
    rot = direction_roll_to_matrix((vx, vy, vz), roll=roll_rad)
    roll, pitch, yaw = pin.rpy.matrixToRpy(rot)
    return x, y, z, roll, pitch, yaw


def _print_result(x, y, z, roll, pitch, yaw):
    print(f"x={x:.6f} y={y:.6f} z={z:.6f} roll={roll:.6f} pitch={pitch:.6f} yaw={yaw:.6f}")


def main():
    if len(sys.argv) > 1:
        vals = [float(v) for v in sys.argv[1:]]
        if len(vals) != 6:
            print("Es werden genau 6 Werte benoetigt: x y z vx vy vz", file=sys.stderr)
            sys.exit(1)
        _print_result(*vec_to_rpy(*vals))
        return

    print("Eingabe: x y z vx vy vz  (q/quit/exit zum Beenden)")
    while True:
        try:
            line = input("> ").strip().lower()
        except EOFError:
            break
        if line in ("q", "quit", "exit", ""):
            break
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            print("Ungueltige Eingabe")
            continue
        if len(vals) != 6:
            print("Es werden genau 6 Werte benoetigt: x y z vx vy vz")
            continue
        _print_result(*vec_to_rpy(*vals))


if __name__ == "__main__":
    main()
