#!/usr/bin/env python3
"""Spielt eine kompilierte q-JSON (Gelenkwinkel) in MeshCat ab -- kein IK, kein CLIK.

Entspricht exakt dem, was play.py am echten Roboter ausfuehrt (Min-Jerk, gleiche Dauer).
Zum Verifizieren einer _q.json vor der Uebertragung auf den Pi.

Nutzung:
    uv run python example/sim/play_sim.py example/poses_body1_pick_paint_drop_q.json
    uv run python example/sim/play_sim.py example/poses_body1_pick_paint_drop_q.json --scale 2.0
    uv run python example/sim/play_sim.py example/poses_body1_pick_paint_drop_q.json --names pick_pre_1 pick_rdy_1
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics import get_end_effector_frame_id
from example.sim.visualizer import Visualizer
import example.sim.traj_sim as _sim

MIN_DURATION = 0.5   # Sekunden, wie in play.py
FALLBACK_SPEED = 0.5  # rad/s, wie play.py --speed default


def load_poses(path: Path) -> list:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["poses"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("poses_file", type=Path, help="Kompilierte q-JSON")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Zeitdehnungsfaktor (>1 = langsamer). Standard: 1.0")
    parser.add_argument("--speed", type=float, default=FALLBACK_SPEED,
                        help="Fallback-Gelenkgeschwindigkeit [rad/s]. Standard: 0.5")
    parser.add_argument("--names", nargs="+", help="Nur diese Posen abspielen")
    parser.add_argument("--prefix", help="Nur Posen mit diesem Namens-Praefix")
    args = parser.parse_args()

    def _on_sigint(sig, frame):
        _sim.should_exit = True
    signal.signal(signal.SIGINT, _on_sigint)

    poses = load_poses(args.poses_file)

    if args.names:
        by_name = {p["name"]: p for p in poses if "name" in p}
        poses = [by_name[n] for n in args.names if n in by_name]
    elif args.prefix:
        poses = [p for p in poses if p.get("name", "").startswith(args.prefix)]

    if not poses:
        print("Keine passenden Posen.", file=sys.stderr)
        sys.exit(1)

    missing_q = [p.get("name", f"#{i}") for i, p in enumerate(poses, 1) if "q" not in p]
    if missing_q:
        print(f"FEHLER: Posen ohne 'q': {missing_q}", file=sys.stderr)
        print("Erst compile_q.py ausfuehren.", file=sys.stderr)
        sys.exit(1)

    print(f"Lade MeshCat...")
    viz = Visualizer(open_browser=True)
    model = viz.model
    end_frame_id = get_end_effector_frame_id(model)
    print(f"{len(poses)} Posen aus {args.poses_file}\n")
    viz.update(pin.neutral(model))

    dt = 1.0 / 50.0
    q_last = pin.neutral(model).copy()
    total = len(poses)

    for idx, pose in enumerate(poses, start=1):
        if _sim.should_exit:
            break
        name = pose.get("name", f"#{idx}")
        q_target = np.array(pose["q"], dtype=float)

        pose_speed = pose.get("speed", args.speed)
        dist = float(np.max(np.abs(q_target - q_last[:len(q_target)])))
        duration = max(MIN_DURATION, dist / pose_speed) * args.scale

        print(f"[{idx}/{total}] {name}  speed={pose_speed} rad/s  duration={duration:.2f}s")
        n_pts = max(2, int(duration / dt))
        s = np.linspace(0.0, 1.0, n_pts)
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        q_full = q_last.copy()
        q_full[:len(q_target)] = q_target

        for b in blend:
            if _sim.should_exit:
                break
            viz.update(q_last + (q_full - q_last) * b)
            time.sleep(dt)
        q_last = q_full

    print("\nFertig.")
    while not _sim.should_exit:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
