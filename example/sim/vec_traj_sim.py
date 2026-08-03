#!/usr/bin/env python3
"""x y z vx vy vz -> vec_to_rpy.py -> roll pitch yaw -> traj_sim.py (Simulation).

Schritt 1: Position + Greifer-Richtungsvektor werden ueber vec_to_rpy.vec_to_rpy()
           in x y z roll pitch yaw umgerechnet (roll=0, siehe vec_to_rpy.py).
Schritt 2: Diese Pose wird als Eingabezeile an ein einmalig gestartetes,
           dauerhaft laufendes traj_sim.py uebergeben, das die Trajektorie
           plant, per CLIK abfaehrt und in MeshCat abspielt. traj_sim.py wird
           NICHT automatisch beendet -- danach koennen weitere Posen direkt
           im traj_sim.py-Prompt (x y z roll pitch yaw, 'q' zum Beenden)
           eingegeben werden.

Nutzung:
    uv run python example/sim/vec_traj_sim.py x y z vx vy vz
    uv run python example/sim/vec_traj_sim.py            # interaktiv
"""
import subprocess
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR))

from vec_to_rpy import vec_to_rpy  # noqa: E402


def send_pose(proc, x, y, z, vx, vy, vz):
    print("Schritt 1: vec_to_rpy -- Richtungsvektor -> roll/pitch/yaw")
    x, y, z, roll, pitch, yaw = vec_to_rpy(x, y, z, vx, vy, vz)
    print(f"  x={x:.6f} y={y:.6f} z={z:.6f} roll={roll:.6f} pitch={pitch:.6f} yaw={yaw:.6f}\n")

    print("Schritt 2: Pose an traj_sim.py uebergeben")
    proc.stdin.write(f"{x} {y} {z} {roll} {pitch} {yaw}\n")
    proc.stdin.flush()
    print("  -> gesendet.\n")


def main():
    proc = subprocess.Popen(
        [sys.executable, str(SIM_DIR / "traj_sim.py")],
        stdin=subprocess.PIPE, text=True,
    )

    if len(sys.argv) > 1:
        vals = [float(v) for v in sys.argv[1:]]
        if len(vals) != 6:
            print("Es werden genau 6 Werte benoetigt: x y z vx vy vz", file=sys.stderr)
            sys.exit(1)
        send_pose(proc, *vals)
        print("traj_sim.py laeuft weiter. Weitere Posen direkt unten eingeben:")
    else:
        print("Eingabe: x y z vx vy vz  (q/quit/exit zum Beenden)")

    while proc.poll() is None:
        try:
            line = input("> ").strip().lower()
        except EOFError:
            break
        if line in ("q", "quit", "exit"):
            proc.stdin.write("q\n")
            proc.stdin.flush()
            break
        if not line:
            continue
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            print("Ungueltige Eingabe")
            continue
        if len(vals) != 6:
            print("Es werden genau 6 Werte benoetigt: x y z vx vy vz")
            continue
        send_pose(proc, *vals)

    proc.wait()


if __name__ == "__main__":
    main()
