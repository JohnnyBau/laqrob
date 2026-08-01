#!/usr/bin/env python3
"""Spielt eine mit teach.py aufgezeichnete poses.json flüssig ab
(MIT-Modus + Gravitations-Feedforward, SE(3)-Geodaete + CLIK ueber move_to_traj()).

Nutzung:
    uv run python example/playback.py --scale 3.0   # gedrosselter Erstlauf (empfohlen)
    uv run python example/playback.py --scale 1.0   # normales Tempo

Sicherheit: bricht bei IK-Fehler die Sequenz sofort ab (haelt aktuelle Pose,
keine automatische Weiterfahrt); fuehrt am Ende immer ctrl.end() aus, das den
Arm per Safe-Home sanft zurueckfaehrt.

Hinweis: move_to_traj() berechnet die komplette CLIK-Gelenktrajektorie VORAB
blockierend in reinem Python (siehe trajectory/clik_tracker.py) -- bevor sich
der Arm ueberhaupt bewegt. Auf dem Pi kann das je nach --dt/--scale spuerbar
dauern; das ist kein Haenger. --dt vergroessern reduziert die Punktzahl und
damit die Vorausberechnungszeit.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose

from _common import (
    load_poses, POSES_FILE, GRIPPER_TAU_OPEN, GRIPPER_TAU_CLOSE, gripper_wait_for_event,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # Pi ist headless
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except ImportError:
    HAVE_PLOT = False


def _plot_log(t, q_target, q_actual) -> Path:
    t = np.array(t)
    q_target = np.array(q_target)
    q_actual = np.array(q_actual)
    # _qd_target wird von move_to_traj()/_send_loop() nie befuellt (bleibt 0) --
    # Ist/Soll-Geschwindigkeit daher aus der Positionsspur ableiten.
    qd_target = np.gradient(q_target, t, axis=0)
    qd_actual = np.gradient(q_actual, t, axis=0)

    n = qd_target.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, qd_target[:, i], label="Soll (aus q_target)")
        ax.plot(t, qd_actual[:, i], label="Ist (gemessen)")
        ax.set_ylabel(f"joint{i + 1} [rad/s]")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Zeit [s]")
    fig.tight_layout()

    out_dir = POSES_FILE.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{int(time.time())}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(poses_path: Path, scale: float, dt: float) -> None:
    poses = load_poses(poses_path)
    if not poses:
        print(f"Keine Posen in {poses_path} gefunden.")
        return

    rebotarm = RebotArm()
    ctrl = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True, dt=dt)
    ctrl.start()

    log_t: list[float] = []
    log_q_target: list[np.ndarray] = []
    log_q_actual: list[np.ndarray] = []
    t0 = time.monotonic()
    aborted = False

    try:
        for pose in poses:
            duration = pose.get("duration", 2.0) * scale
            n_pts = int(duration / dt)
            print(f"-> {pose['name']}  (duration={duration:.2f}s, scale={scale}, "
                  f"dt={dt}, ~{n_pts} CLIK-Punkte)")
            print("   berechne Trajektorie (CLIK, blockierend) ...", flush=True)

            t_calc0 = time.monotonic()
            if "q" in pose:
                ok = ctrl.move_to_q_traj(np.array(pose["q"]), duration=duration)
            else:
                ok = ctrl.move_to_traj(
                    x=pose["x"], y=pose["y"], z=pose["z"],
                    roll=pose["roll"], pitch=pose["pitch"], yaw=pose["yaw"],
                    duration=duration,
                )
            print(f"   Vorausberechnung fertig nach {time.monotonic() - t_calc0:.2f}s"
                  f" -- Bewegung startet jetzt.")

            if not ok:
                print(f"   ABBRUCH: IK/Trajektorie fehlgeschlagen bei '{pose['name']}'. "
                      f"Sequenz gestoppt, aktuelle Pose wird gehalten.")
                aborted = True
                break

            while ctrl._moving:
                pos, vel, _ = rebotarm.get_state(request_feedback=False)
                log_t.append(time.monotonic() - t0)
                log_q_target.append(ctrl._q_target.copy())
                log_q_actual.append(pos[: ctrl._n].copy())
                time.sleep(0.02)

            gripper = pose.get("gripper", "none")
            if gripper in ("open", "close"):
                if gripper == "open":
                    ctrl.open_gripper(tau=GRIPPER_TAU_OPEN)
                else:
                    ctrl.close_gripper(tau=GRIPPER_TAU_CLOSE)
                # Regelkreis (ctrl._loop_cb) sendet die Kraft bereits kontinuierlich,
                # hier nur beobachten bis Widerstand/max-travel/timeout.
                event, g_pos, travel = gripper_wait_for_event(rebotarm, send=False)
                if event == "resistance":
                    print(f"   Greifer: Widerstand bei pos={g_pos:+.4f} rad (travel={travel:.3f} rad).")
                elif event == "max_travel":
                    print(f"   Greifer: kein Widerstand bis max-travel gefunden "
                          f"(pos={g_pos:+.4f} rad) -- vermutlich kein Objekt.")
                else:
                    print(f"   Greifer: Timeout bei pos={g_pos:+.4f} rad.")
            time.sleep(0.2)  # kurze Ruhephase an jedem Wegpunkt
    finally:
        ctrl.end()  # sicheres Safe-Home, unabhaengig von Erfolg/Abbruch
        print("Wiedergabe abgebrochen." if aborted else "Wiedergabe beendet.")
        if HAVE_PLOT and len(log_t) > 2:
            out_path = _plot_log(log_t, log_q_target, log_q_actual)
            print(f"Geschwindigkeits-Plot gespeichert: {out_path}")
        elif not HAVE_PLOT:
            print("matplotlib nicht verfuegbar, kein Plot erzeugt.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, default=POSES_FILE, help="Pfad zur poses.json")
    parser.add_argument(
        "--scale", type=float, default=3.0,
        help="Multipliziert alle duration-Werte (>1 = gedrosselter/sichererer Lauf, 1.0 = Originaltempo)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02,
        help="Zeitschritt der Trajektorien-Abtastung in s (kleiner = mehr CLIK-Punkte = "
             "laengere Vorausberechnung, aber feiner abgetastet). Standard 0.02 (50Hz).",
    )
    args = parser.parse_args()
    run(args.poses, args.scale, args.dt)


if __name__ == "__main__":
    main()
