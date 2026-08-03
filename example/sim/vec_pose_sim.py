#!/usr/bin/env python3
"""Interaktive Pose-Definition ueber Werkzeugrichtung + Rollwinkel + Greiferoeffnung.

Statt x/y/z/roll/pitch/yaw direkt anzugeben, wird die Zielorientierung hier
ueber die Werkzeugrichtung (Vektor vx/vy/vz, = lokale X-Achse des Endeffektors,
siehe live_pose_plot.py) plus einen Rollwinkel um diese Achse festgelegt.
Das Skript rechnet das in x/y/z/roll/pitch/yaw um (kompatibel mit
poses.json/poses_fixed.json), loest die IK und zeigt das Ergebnis live in
MeshCat -- so laesst sich pruefen, ob die eingegebenen Werte zur gewuenschten
Pose fuehren, bevor sie z.B. in poses_fixed.json uebernommen werden.

Eingabe (8 Werte, durch Leerzeichen getrennt):
    x y z vx vy vz greifer roll

    x y z     -- Zielposition, Meter
    vx vy vz  -- Werkzeugrichtung (muss nicht normiert sein)
    greifer   -- Greifer-Gelenkwinkel, Radiant (wie "gripper_pos" in poses.json)
    roll      -- Rollwinkel um die Werkzeugachse, Radiant

Nutzung:
    uv run python example/sim/vec_pose_sim.py

Steuerung:
    q/quit/exit: Beenden
"""
import sys
import signal
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics.inverse_kinematics import (
    compute_ik,
    direction_roll_to_matrix,
)
from example.sim.visualizer import Visualizer

should_exit = False


def signal_handler(sig, frame):
    global should_exit
    should_exit = True


def main():
    signal.signal(signal.SIGINT, signal_handler)

    print("Lade Visualisierer...")
    viz = Visualizer()
    viz.neutral()

    print("MeshCat bereit. Eingabe: x y z vx vy vz greifer roll  (roll in Rad)")
    print("q/quit/exit: Beenden\n")

    while not should_exit:
        time.sleep(0.01)

        try:
            line = input("pose(x y z vx vy vz greifer roll) > ").strip().lower()
        except EOFError:
            break

        if line in ("q", "quit", "exit", ""):
            break

        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            print("Ungueltige Eingabe\n")
            continue

        if len(vals) != 8:
            print("Es werden genau 8 Werte benoetigt: x y z vx vy vz greifer roll\n")
            continue

        x, y, z, vx, vy, vz, greifer, roll = vals
        target_pos = np.array([x, y, z])
        rot = direction_roll_to_matrix((vx, vy, vz), roll)
        rpy = pin.rpy.matrixToRpy(rot)

        result = compute_ik(None, target_pos, rot)

        # compute_ik loest ueber das volle Modell (Arm + Greifer); die Greifer-
        # Spalte der Jacobi-Matrix ist fuer die Endeffektor-Pose irrelevant,
        # daher hier explizit mit der gewuenschten Greiferoeffnung ueberschreiben.
        q_full = result.q.copy()
        q_full[-1] = greifer
        q_arm = q_full[:-1]
        viz.update(q_full)

        status = "konvergiert" if result.success else "NICHT konvergiert"
        print(f"  IK {status}  (err={result.error:.3e}, iter={result.iterations})")
        print(f"  -> x={x:.6f} y={y:.6f} z={z:.6f}")
        print(f"     roll={rpy[0]:.6f} pitch={rpy[1]:.6f} yaw={rpy[2]:.6f}")
        print(f"     gripper_pos={greifer:.6f}")
        print(f"     q      = {[round(v, 6) for v in q_arm.tolist()]}")
        print(f"     q_full = {[round(v, 6) for v in q_full.tolist()]}\n")


if __name__ == "__main__":
    main()
