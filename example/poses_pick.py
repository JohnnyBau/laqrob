#!/usr/bin/env python3
"""Erzeugt die reinen Greifer-POSITIONEN fuer eine Rasterentnahme aus einer Eurobox.

Aus einer einzigen angelernten Referenzpose (Teil UNTEN LINKS im Raster,
x y z + Greifer-Richtungsvektor vx vy vz = Eintauchachse = Normalenvektor der
Kistenebene) sowie Rasterparametern (nx, ny, dx, dy, Eintauchtiefe) werden fuer
jedes Teil 4 benannte Positionen berechnet (Schema x/y/z/vx/vy/vz, wie
poses_v.json) und in eine JSON-Datei geschrieben:

    <prefix>_pre_n    Standoff (Referenzpose - Eintauchtiefe entlang der
                      Greiferachse)
    <prefix>_rdy_n    Eintauchen (Greifhoehe, angelernte Tiefe)
    <prefix>_open_n   Greifer oeffnen (= Teil greifen, spreizender
                      Innengreifer), gleiche Position wie *_rdy_n
    <prefix>_post_n   Herausziehen (= Standoff, identisch zu *_pre_n)

Es findet KEINE Winkel-/IK-Berechnung statt (kein roll/pitch/yaw, kein q, kein
duration/gripper-Feld) -- reine Cartesian-Positionen + Richtungsvektor. Die
Umwandlung in roll/pitch/yaw uebernimmt vec_to_rpy.py, die Simulation
vec_traj_sim.py/poses_traj_sim.py.

Kisten-Ausrichtung: der Richtungsvektor (vx,vy,vz) ist der Normalenvektor der
Kistenebene (= Eintauchachse). Die Rasterachsen liegen in der dazu senkrechten
Ebene: lokale Y-Achse = nx-Richtung, lokale Z-Achse = ny-Richtung (gleiche
Konvention wie direction_roll_to_matrix()/vec_to_rpy.py, roll=0).

Reihenfolge: zeilenweise, n=1..nx*ny (erst alle nx in Reihe j=0, dann j=1, ...).

Ausgabedatei: example/poses_<partname>_pick.json (partname als Parameter).

Nutzung:
    uv run python example/poses_pick.py bolt_m6 0.2 -0.36 0.17 0.0 -0.701 -0.701 \\
        --dive-depth 0.03 --nx 4 --ny 3 --dx 0.05 --dy 0.06
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reBotArm_control_py.kinematics.inverse_kinematics import direction_roll_to_matrix

POSES_DIR = Path(__file__).resolve().parent


def load_poses(path: Path) -> tuple[list, dict]:
    """Gibt (poses, meta) zurueck; meta enthaelt z.B. {'speed': 0.4} aus dem Wrapper."""
    if not path.exists():
        return [], {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, {}
    return data["poses"], {k: v for k, v in data.items() if k != "poses"}


def save_poses(poses: list, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**meta, "poses": poses} if meta else poses
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def build_box_pick_positions(x, y, z, vx, vy, vz, dive_depth, nx, ny, dx, dy,
                              prefix="pick", nx_dir=1, ny_dir=1):
    """Berechnet die Greiferpositionen (x/y/z/vx/vy/vz) je Rasterzelle.

    nx_dir/ny_dir (+1/-1): kehrt die Rasterrichtung der jeweiligen Achse um, falls
    die Kiste entgegen der aus direction_roll_to_matrix abgeleiteten lokalen
    Y/Z-Achse liegt.
    """
    rot = direction_roll_to_matrix((vx, vy, vz), roll=0.0)
    dive_axis = rot[:, 0]
    grid_x, grid_y = nx_dir * rot[:, 1], ny_dir * rot[:, 2]
    direction = (float(vx), float(vy), float(vz))
    pos_ref = np.array([x, y, z])

    def _pose(name, pos):
        return {
            "name": name, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "vx": direction[0], "vy": direction[1], "vz": direction[2],
        }

    poses = []
    n = 0
    for j in range(ny):
        for i in range(nx):
            n += 1
            pos_rdy = pos_ref + i * dx * grid_x + j * dy * grid_y
            pos_pre = pos_rdy - dive_depth * dive_axis
            poses.append(_pose(f"{prefix}_pre_{n}", pos_pre))
            poses.append(_pose(f"{prefix}_rdy_{n}", pos_rdy))
            poses.append(_pose(f"{prefix}_open_{n}", pos_rdy))
            poses.append(_pose(f"{prefix}_post_{n}", pos_pre))
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("partname", help="Teilname, bestimmt die Ausgabedatei poses_<partname>_pick.json")
    parser.add_argument("x", type=float, help="Referenzposition Teil unten links, X [m]")
    parser.add_argument("y", type=float, help="Referenzposition Teil unten links, Y [m]")
    parser.add_argument("z", type=float, help="Referenzposition Teil unten links, Z [m]")
    parser.add_argument("vx", type=float, help="Greifer-Richtungsvektor X (Eintauchachse)")
    parser.add_argument("vy", type=float, help="Greifer-Richtungsvektor Y")
    parser.add_argument("vz", type=float, help="Greifer-Richtungsvektor Z")
    parser.add_argument("--dive-depth", type=float, required=True, help="Eintauchtiefe [m]")
    parser.add_argument("--nx", type=int, required=True, help="Teile in Kisten-X-Richtung (Draufsicht)")
    parser.add_argument("--ny", type=int, required=True, help="Teile in Kisten-Y-Richtung (Draufsicht)")
    parser.add_argument("--dx", type=float, required=True, help="Abstand zwischen Teilen in Kisten-X [m]")
    parser.add_argument("--dy", type=float, required=True, help="Abstand zwischen Teilen in Kisten-Y [m]")
    parser.add_argument("--nx-dir", type=int, choices=[1, -1], default=-1,
                         help="Vorzeichen der Kisten-X-Rasterrichtung (-1 = umgekehrt, Standard)")
    parser.add_argument("--ny-dir", type=int, choices=[1, -1], default=1,
                         help="Vorzeichen der Kisten-Y-Rasterrichtung (-1 = umgekehrt)")
    parser.add_argument("--prefix", default="pick", help="Namenspraefix (Standard 'pick' -> pick_pre_1, ...)")
    parser.add_argument("--speed", type=float, default=None,
                         help="Pick-Geschwindigkeit [rad/s] fuer play.py (wird in der JSON gespeichert)")
    parser.add_argument("--poses", type=Path, default=None,
                         help="Ziel-JSON-Datei (Standard: poses_<partname>_pick.json)")
    args = parser.parse_args()
    poses_path = args.poses if args.poses is not None else POSES_DIR / f"poses_{args.partname}_pick.json"

    print(f"Schritt 1: Rasterpositionen berechnen ({args.nx}x{args.ny} Teile, "
          f"Eintauchtiefe={args.dive_depth} m)")
    new_poses = build_box_pick_positions(
        args.x, args.y, args.z, args.vx, args.vy, args.vz, args.dive_depth,
        args.nx, args.ny, args.dx, args.dy, args.prefix,
        args.nx_dir, args.ny_dir,
    )
    print(f"  -> {len(new_poses)} Positionen erzeugt ({args.nx * args.ny} Teile x 4 Schritte).\n")

    print(f"Schritt 2: in {poses_path} speichern")
    existing, meta = load_poses(poses_path)
    if args.speed is not None:
        meta["speed"] = args.speed
    new_names = {p["name"] for p in new_poses}
    kept = [p for p in existing if p.get("name") not in new_names]
    n_replaced = len(existing) - len(kept)
    save_poses(kept + new_poses, poses_path, meta)
    print(f"  -> {len(new_poses)} Positionen geschrieben ({n_replaced} vorhandene mit gleichem Namen ersetzt), "
          f"Gesamt in Datei: {len(kept) + len(new_poses)}.")


if __name__ == "__main__":
    main()
