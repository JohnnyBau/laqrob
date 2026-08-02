#!/usr/bin/env python3
"""Rasterentnahme aus einer (ggf. 45 Grad gedrehten) Eurobox.

Eine einzige angelernte Referenzpose (Teil OBEN LINKS im Raster, siehe
teach.py/live_pose_plot.py) + Rasterparameter (nx, ny, dx, dy, Kistenwinkel)
ergeben alle Greifposen. Orientierung (roll/pitch/yaw) bleibt fuer alle Teile
gleich (aus der Referenzpose uebernommen) -- nur x/y/z wandern im Raster.

Greifer-Logik (spreizender Innen-Greifer): OEFFNEN = Teil greifen (Kraft bis
Widerstand/max-Oeffnung/Timeout), SCHLIESSEN = loslassen. Ablage aktuell nur
Platzhalter an der Nullstellung -- spaeter durch Lackier-/Ablageprozess ersetzt.

Zyklus pro Zelle:
    Nullstellung -> pos1 (Standoff, Eintauchtiefe VOR dem Teil)
                 -> pos2 (Eintauchen, Greifhoehe)
                 -> greifen (oeffnen bis Widerstand/max-Oeffnung/Timeout)
                 -> pos1 (herausziehen)
                 -> Nullstellung (safe_home)
                 -> loslassen (schliessen, Platzhalter)
                 -> naechste Zelle (zeilenweise: erst alle nx in einer Reihe)

Nutzung:
    # 1) Dry-Run: nur MeshCat-Simulation, prueft Erreichbarkeit alller Zellen,
    #    keine Hardware-Verbindung. IMMER ZUERST HIERMIT PRUEFEN.
    uv run python example/pick_grid.py --ref pose_2 --nx 4 --ny 3 \\
        --dx 0.05 --dy 0.06 --box-angle 45 --dive-depth 0.03 --dry-run

    # 2) Echter Lauf (nach visueller Pruefung!), gedrosselt per --scale
    uv run python example/pick_grid.py --ref pose_2 --nx 4 --ny 3 \\
        --dx 0.05 --dy 0.06 --box-angle 45 --dive-depth 0.03 \\
        --grip-force 1.0 --max-gripper-open 2.5 --scale 3.0

Sicherheit: bricht bei IK-Fehlschlag einer Zelle die GESAMTE Sequenz sofort ab
(haelt Pose, kein automatisches Weiterfahren); fuehrt am Ende immer ctrl.end()
aus (Safe-Home + disconnect).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pinocchio as pin

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import (
    compute_fk,
    get_end_effector_frame_id,
    pos_rot_to_se3,
    solve_ik_with_retry,
    IKSolverParams,
)

from _common import (
    load_poses, POSES_FILE, GRIPPER_MAX_TRAVEL, GRIPPER_TIMEOUT,
    gripper_wait_for_event,
)


def _find_ref_pose(poses: list[dict], name: str) -> dict:
    for p in poses:
        if p.get("name") == name:
            return p
    raise SystemExit(f"Referenzpose '{name}' nicht in poses.json gefunden.")


def compute_grid_cells(
    ref: dict, nx: int, ny: int, dx: float, dy: float,
    box_angle_deg: float, dive_depth: float,
) -> list[dict]:
    """Baut fuer jede Rasterzelle pos1 (Standoff) + pos2 (Eintauchen), zeilenweise sortiert.

    Raster liegt in der Welt-XY-Ebene (Draufsicht), um box_angle_deg um Welt-Z
    gegen die Roboter-Weltachsen gedreht. Referenzzelle (i=0, j=0) = die
    angelernte Pose (Teil oben links) selbst. Eintauchachse = lokale
    Werkzeugachse (Greifer-Blickrichtung) der Referenzpose, nicht Welt-Z.
    """
    ref_pos2 = np.array([ref["x"], ref["y"], ref["z"]])
    roll, pitch, yaw = ref["roll"], ref["pitch"], ref["yaw"]

    theta = np.radians(box_angle_deg)
    grid_x = np.array([np.cos(theta), np.sin(theta), 0.0])   # +i Richtung (nx)
    grid_y = np.array([-np.sin(theta), np.cos(theta), 0.0])  # +j Richtung (ny)

    R_ref = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    dive_axis = R_ref @ np.array([1.0, 0.0, 0.0])
    dive_axis /= np.linalg.norm(dive_axis)

    cells = []
    for j in range(ny):        # Reihe
        for i in range(nx):    # Spalte innerhalb der Reihe (zeilenweise)
            pos2 = ref_pos2 + i * dx * grid_x + j * dy * grid_y
            pos1 = pos2 - dive_depth * dive_axis
            cells.append({
                "i": i, "j": j,
                "pos1": pos1, "pos2": pos2,
                "roll": roll, "pitch": pitch, "yaw": yaw,
            })
    return cells


def _wait_move(ctrl: RebotArmEndPose) -> None:
    while ctrl._moving:
        time.sleep(0.02)


def run_dry(cells: list[dict], dt: float) -> None:
    """MeshCat-only: prueft IK-Erreichbarkeit aller pos1/pos2 und animiert das Raster."""
    from example.sim.visualizer import Visualizer

    print("Lade MeshCat-Visualisierer...")
    viz = Visualizer()
    model = viz.model
    data = model.createData()
    end_frame_id = get_end_effector_frame_id(model)
    ik_params = IKSolverParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

    q = pin.neutral(model).copy()
    viz.update(q)

    n_fail = 0
    for cell in cells:
        for key in ("pos1", "pos2"):
            target = pos_rot_to_se3(
                cell[key], roll=cell["roll"], pitch=cell["pitch"], yaw=cell["yaw"],
            )
            ik_result = solve_ik_with_retry(model, data, end_frame_id, target, q.copy(), ik_params)
            label = f"Zelle({cell['i']},{cell['j']}) {key}"
            if not ik_result.success:
                n_fail += 1
                print(f"  [FEHLER] {label}: IK nicht konvergiert (err={ik_result.error:.3e}) "
                      f"-- Kistenwinkel/Eintauchtiefe pruefen!")
                continue
            print(f"  [OK] {label}: q gefunden.")
            q = ik_result.q
            viz.update(q)
            time.sleep(dt)

    print(f"\nDry-Run fertig: {len(cells) * 2 - n_fail}/{len(cells) * 2} Zielposen erreichbar.")
    if n_fail:
        print("ACHTUNG: nicht alle Zielposen erreichbar -- NICHT auf echter Hardware starten, "
              "bevor das behoben ist (z.B. --box-angle Vorzeichen umdrehen).")
    else:
        print("Alle Zielposen erreichbar. Raster kann auf echter Hardware getestet werden "
              "(empfohlen: --scale 3.0 fuer den ersten Lauf).")


def run_hardware(
    cells: list[dict], travel_duration: float, dive_duration: float, scale: float,
    grip_force: float, release_force: float, max_gripper_open: float, grip_timeout: float,
) -> None:
    rebotarm = RebotArm()
    ctrl = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True)
    ctrl.start()

    aborted = False
    n_found = 0
    try:
        for cell in cells:
            label = f"Zelle({cell['i']},{cell['j']})"
            print(f"-> {label}: Standoff anfahren")
            if not ctrl.move_to_traj(*cell["pos1"], cell["roll"], cell["pitch"], cell["yaw"],
                                      duration=travel_duration * scale):
                print(f"   ABBRUCH: IK/Trajektorie fehlgeschlagen bei {label} (pos1).")
                aborted = True
                break
            _wait_move(ctrl)

            print(f"   {label}: eintauchen")
            if not ctrl.move_to_traj(*cell["pos2"], cell["roll"], cell["pitch"], cell["yaw"],
                                      duration=dive_duration * scale):
                print(f"   ABBRUCH: IK/Trajektorie fehlgeschlagen bei {label} (pos2).")
                aborted = True
                break
            _wait_move(ctrl)

            ctrl.open_gripper(tau=-abs(grip_force))
            event, g_pos, travel = gripper_wait_for_event(
                rebotarm, send=False, max_travel=max_gripper_open, timeout=grip_timeout,
            )
            if event == "resistance":
                n_found += 1
                print(f"   {label}: Teil gegriffen (pos={g_pos:+.4f} rad, travel={travel:.3f} rad).")
            else:
                print(f"   {label}: kein Teil gefunden ({event}, pos={g_pos:+.4f} rad) -- ueberspringe.")

            print(f"   {label}: herausziehen")
            if not ctrl.move_to_traj(*cell["pos1"], cell["roll"], cell["pitch"], cell["yaw"],
                                      duration=dive_duration * scale):
                print(f"   ABBRUCH: IK/Trajektorie fehlgeschlagen bei {label} (Rueckzug).")
                aborted = True
                break
            _wait_move(ctrl)

            print(f"   {label}: Nullstellung")
            ctrl.safe_home()

            print(f"   {label}: loslassen (Platzhalter -- spaeter Lackier-/Ablageprozess)")
            ctrl.close_gripper(tau=abs(release_force))
            gripper_wait_for_event(
                rebotarm, send=False, max_travel=max_gripper_open, timeout=grip_timeout,
            )
    finally:
        ctrl.end()  # sicheres Safe-Home + disconnect, unabhaengig von Erfolg/Abbruch
        status = "abgebrochen" if aborted else "beendet"
        print(f"\nRasterentnahme {status}. {n_found}/{len(cells)} Teile gegriffen.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=Path, default=POSES_FILE, help="Pfad zur poses.json")
    parser.add_argument("--ref", required=True, help="Name der angelernten Referenzpose (Teil oben links)")
    parser.add_argument("--nx", type=int, required=True, help="Anzahl Teile in x-Richtung (Draufsicht)")
    parser.add_argument("--ny", type=int, required=True, help="Anzahl Teile in y-Richtung (Draufsicht)")
    parser.add_argument("--dx", type=float, required=True, help="Abstand zwischen Teilen in x [m]")
    parser.add_argument("--dy", type=float, required=True, help="Abstand zwischen Teilen in y [m]")
    parser.add_argument("--box-angle", type=float, default=45.0,
                         help="Drehwinkel der Kiste gegenueber dem Roboter [Grad], signiert")
    parser.add_argument("--dive-depth", type=float, required=True,
                         help="Eintauchtiefe entlang der Greifer-Werkzeugachse [m]")
    parser.add_argument("--grip-force", type=float, default=1.0, help="Griffkraft (Oeffnen) [Nm]")
    parser.add_argument("--release-force", type=float, default=1.0, help="Loslass-Kraft (Schliessen) [Nm]")
    parser.add_argument("--max-gripper-open", type=float, default=GRIPPER_MAX_TRAVEL,
                         help="Maximale Greifferoeffnung [rad] -- kein Widerstand bis hierhin = kein Teil")
    parser.add_argument("--grip-timeout", type=float, default=GRIPPER_TIMEOUT, help="Timeout Greifvorgang [s]")
    parser.add_argument("--travel-duration", type=float, default=2.0,
                         help="Dauer Nullstellung<->Standoff [s] (vor --scale)")
    parser.add_argument("--dive-duration", type=float, default=3.0,
                         help="Dauer Eintauchen/Herausziehen [s] (vor --scale, langsamer als travel)")
    parser.add_argument("--scale", type=float, default=3.0,
                         help="Multipliziert alle duration-Werte (>1 = gedrosselter/sichererer Lauf)")
    parser.add_argument("--dt", type=float, default=0.05, help="Schrittzeit fuer Dry-Run-Animation [s]")
    parser.add_argument("--dry-run", action="store_true",
                         help="Nur MeshCat-Simulation (keine Hardware) -- IMMER ZUERST verwenden")
    args = parser.parse_args()

    poses = load_poses(args.poses)
    ref = _find_ref_pose(poses, args.ref)
    cells = compute_grid_cells(ref, args.nx, args.ny, args.dx, args.dy, args.box_angle, args.dive_depth)
    print(f"{len(cells)} Rasterzellen berechnet ({args.nx}x{args.ny}, Winkel={args.box_angle} Grad).")

    if args.dry_run:
        run_dry(cells, args.dt)
    else:
        run_hardware(
            cells, args.travel_duration, args.dive_duration, args.scale,
            args.grip_force, args.release_force, args.max_gripper_open, args.grip_timeout,
        )


if __name__ == "__main__":
    main()
