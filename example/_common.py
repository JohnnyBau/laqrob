"""Gemeinsame Hilfsfunktionen fuer teach.py / playback.py.

Eigene, additive Datei -- kein Bestandteil des SeeedStudio-Repos, importiert
nur dessen oeffentliche API. Es werden keine vorhandenen Repo-Dateien veraendert.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import compute_generalized_gravity

# Posen im Repo ablegen (example/poses.json), damit teach.py/live_pose_plot.py/
# playback.py immer dieselbe, versionierte Datei verwenden.
POSES_DIR = Path(__file__).resolve().parent
POSES_FILE = POSES_DIR / "poses.json"

GRIPPER_ACTIONS = ("open", "close", "none")

# Greifer = Motor 0x07 (DM4310, Tmax=10 Nm laut motorbridge DAMIAO_MODEL_LIMITS).
# ctrl.open_gripper(tau)/close_gripper(tau) regeln reine Kraft (kp=0).
# Vorzeichen am echten Arm kalibriert (example/gripper_test.py): negativ = oeffnen.
GRIPPER_TAU_OPEN = -1.0
GRIPPER_TAU_CLOSE = 1.0

# Parameter fuer gripper_wait_for_event() (Stillstands-/Verfahrweg-Erkennung).
GRIPPER_KD_FORCE = 0.3
GRIPPER_STALL_VEL = 0.05
GRIPPER_STALL_DURATION = 0.3
GRIPPER_START_GRACE = 0.3
GRIPPER_MAX_TRAVEL = 3.0
GRIPPER_TIMEOUT = 8.0


def gravity_hold(r: RebotArm, dt: float) -> None:
    """Freilauf-Regler (Gravitationskompensation), analog example/9_gravity_compensation.py."""
    q = r.arm.get_positions(request_feedback=False)
    tau_g = compute_generalized_gravity(q=q)
    r.arm.send_mit(
        pos=q,
        vel=np.zeros(r.arm.num_joints),
        kp=np.full(r.arm.num_joints, 5.0),
        kd=np.full(r.arm.num_joints, 2.0),
        tau=tau_g,
    )
    if r.has_gripper:
        r.gripper.send_mit(r.gripper.get_positions())


def gripper_wait_for_event(
    r: RebotArm,
    send: bool,
    tau: float = 0.0,
    max_travel: float = GRIPPER_MAX_TRAVEL,
    timeout: float = GRIPPER_TIMEOUT,
) -> tuple[str, float, float]:
    """Beobachtet den Greifer bis Widerstand, max_travel oder timeout.

    send=True:  sendet selbst konstante Kraft `tau` jeden Zyklus (Standalone-Test).
    send=False: nur Beobachtung -- Kraft wird extern gesendet (z.B. durch den
                RebotArmEndPose-Regelkreis nach ctrl.open_gripper()/close_gripper()).

    Rueckgabe: (event, position, travel) mit event in
               {"resistance", "max_travel", "timeout"}.
    """
    pos0, _, _ = r.get_state()
    start_pos = pos0[-1]  # Greifer = letztes Gelenk in der Config
    t0 = time.monotonic()
    stall_since: float | None = None

    while True:
        now = time.monotonic() - t0
        pos, vel, _ = r.get_state()
        g_pos, g_vel = pos[-1], vel[-1]
        travel = abs(g_pos - start_pos)

        if send:
            r.gripper.send_mit(
                pos=np.array([g_pos]), vel=np.zeros(1),
                kp=np.zeros(1), kd=np.full(1, GRIPPER_KD_FORCE),
                tau=np.array([tau]),
            )

        if now > GRIPPER_START_GRACE and abs(g_vel) < GRIPPER_STALL_VEL:
            if stall_since is None:
                stall_since = now
            elif now - stall_since >= GRIPPER_STALL_DURATION:
                return "resistance", g_pos, travel
        else:
            stall_since = None

        if travel >= max_travel:
            return "max_travel", g_pos, travel
        if now > timeout:
            return "timeout", g_pos, travel

        time.sleep(0.02)


def load_poses(path: Path = POSES_FILE) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_poses(poses: list[dict], path: Path = POSES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poses, indent=2, ensure_ascii=False))
