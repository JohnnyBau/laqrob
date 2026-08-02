#!/usr/bin/env python3
"""Live-Plot: Endeffektor-Pose (x/y/z/roll + Greifer-Vektor) waehrend Free-Drive.

Arm haengt in Gravitationskompensation (wie teach.py, unveraendert), du
kannst ihn frei mit der Hand fuehren. Die Pose wird per FK live berechnet
und in einem Browser-Plot dargestellt (matplotlib WebAgg-Backend -- kein
Display am Pi noetig).

Ueber den Status-Text (Port 8990) laesst sich die aktuelle Pose per Formular
als benannte Pose in poses.json speichern (gleiches Format wie teach.py) --
anschliessend mit example/playback.py anfahrbar.

Als CLI-Skript:
    uv run python example/live_pose_plot.py
    -> im Browser (vom PC aus) http://<pi-ip>:8989 oeffnen (Plot)
    -> http://<pi-ip>:8990 oeffnen (Status-Text + Pose speichern)
    Strg+C im Terminal zum Beenden.

Als importierbare Funktion (fuer andere Programme -- blockiert, bis das
Fenster geschlossen/Strg+C gedrueckt wird, dann Rueckgabe):
    from example.live_pose_plot import run
    gespeicherte_posen = run()  # Liste von Posen dieser Sitzung, inkl. Greiferposition
"""
import html
import http.server
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8989
matplotlib.rcParams["webagg.open_in_browser"] = False

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.kinematics import joint_to_pose, pad_q_for_model

from _common import gravity_hold, load_poses, save_poses, POSES_FILE
from example.sim.visualizer import Visualizer

WINDOW_S = 20.0
SAMPLE_HZ = 20.0
N = int(WINDOW_S * SAMPLE_HZ)
STATUS_INTERVAL_S = 10.0
STATUS_HTTP_PORT = 8990

_status_lock = threading.Lock()
_status_holder = {"text": "warte auf erstes Update...", "saved_msg": ""}
_state_lock = threading.Lock()
_latest_state = {"q": None, "pos": None, "rpy": None}
_n_arm = None  # von run() gesetzt
_has_gripper = False  # von run() gesetzt
_session_saves: list[dict] = []  # in dieser Sitzung gespeicherte Posen -- Rueckgabe von run()


def _save_pose(name: str, gripper: str) -> str:
    with _state_lock:
        q, pos, rpy = _latest_state["q"], _latest_state["pos"], _latest_state["rpy"]
    if q is None:
        return "Noch keine Pose verfuegbar -- kurz warten und erneut versuchen."
    poses = load_poses()
    gripper_pos = float(q[_n_arm]) if _has_gripper else None  # Gelenkwinkel des Greifers (letztes Gelenk)
    entry = {
        "name": name if name else f"pose_{len(poses) + 1}",
        "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
        "roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2]),
        "duration": 2.0,
        "gripper": gripper if gripper in ("open", "close") else "none",
        "gripper_pos": gripper_pos,
        # reale Gelenkwinkel: playback.py faehrt darauf direkt (Min-Jerk, keine IK).
        "q": [float(v) for v in q[:_n_arm]],
    }
    poses.append(entry)
    save_poses(poses)
    with _state_lock:
        _session_saves.append(dict(entry))
    return f"gespeichert: '{entry['name']}' ({len(poses)} Posen in {POSES_FILE})"


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 -- von http.server vorgegeben
        parsed = urlparse(self.path)
        if parsed.path == "/save":
            qs = parse_qs(parsed.query)
            name = qs.get("name", [""])[0].strip()
            gripper = qs.get("gripper", ["none"])[0]
            with _status_lock:
                _status_holder["saved_msg"] = _save_pose(name, gripper)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return

        with _status_lock:
            text = _status_holder["text"]
            saved_msg = _status_holder["saved_msg"]
        saved_html = f"<p style='color:green'>{html.escape(saved_msg)}</p>" if saved_msg else ""
        body = (
            "<html><head><meta http-equiv='refresh' content='2'>"
            "<meta charset='utf-8'></head><body>"
            f"<pre style='font-size:18px'>{text}</pre>"
            "<form action='/save' method='get'>"
            "Name: <input name='name' type='text'> "
            "Greifer: <select name='gripper'>"
            "<option value='none'>keine Aktion</option>"
            "<option value='open'>oeffnen</option>"
            "<option value='close'>schliessen</option>"
            "</select> "
            "<button type='submit'>Pose speichern</button>"
            "</form>"
            f"{saved_html}"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keine Konsolenausgabe pro Request
        pass


def _start_status_http_server() -> None:
    server = http.server.ThreadingHTTPServer(("0.0.0.0", STATUS_HTTP_PORT), _StatusHandler)
    server.serve_forever()

LABELS = ["x [m]", "y [m]", "z [m]", "roll [deg]"]
VECTOR_LABEL = "Greifer-Vektor [-]"
VECTOR_NAMES = ["vx", "vy", "vz"]


def run() -> list[dict]:
    """Startet Live-Plot + Simulation, blockiert bis Fensterschluss/Strg+C.

    Rueckgabe: Liste der in dieser Sitzung ueber das Speichern-Formular
    gesicherten Posen (x/y/z/roll/pitch/yaw/q/gripper/gripper_pos) -- fuer
    weitere Programmaufrufe (z.B. direkte Weiterverarbeitung ohne poses.json).
    """
    global _n_arm, _has_gripper
    _session_saves.clear()
    rebotarm = RebotArm()
    rebotarm.connect()
    rebotarm.arm.mode_mit()
    if rebotarm.has_gripper:
        rebotarm.gripper.mode_mit()
    rebotarm.enable_all()
    rebotarm.start_control_loop(gravity_hold, rate=rebotarm.rate)
    _n_arm = rebotarm.arm.num_joints
    _has_gripper = rebotarm.has_gripper
    threading.Thread(target=_start_status_http_server, daemon=True).start()
    print(f"Kopierbarer Status-Text + Pose speichern: http://0.0.0.0:{STATUS_HTTP_PORT}")

    viz = Visualizer()
    print(f"3D-Simulation: {viz.meshcat.url().replace('127.0.0.1', '192.168.178.130')}")

    t_buf = deque(maxlen=N)
    val_buf = [deque(maxlen=N) for _ in range(len(LABELS))]
    vector_buf = [deque(maxlen=N) for _ in range(3)]
    t0 = time.time()

    fig, axes = plt.subplots(len(LABELS) + 1, 1, sharex=True, figsize=(8, 10))
    lines = []
    for ax, label in zip(axes, LABELS):
        (line,) = ax.plot([], [])
        ax.set_ylabel(label)
        ax.grid(True)
        lines.append(line)
    vector_axis = axes[-1]
    vector_lines = [vector_axis.plot([], [], label=name)[0] for name in VECTOR_NAMES]
    vector_axis.set_ylabel(VECTOR_LABEL)
    vector_axis.set_ylim(-1.1, 1.1)  # Einheitsvektor -- Bereich ist fest bekannt
    vector_axis.legend(loc="upper right", fontsize=8)
    vector_axis.grid(True)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Live Endeffektor-Pose (Free-Drive) -- Strg+C im Terminal zum Beenden")
    status_text = fig.text(0.5, 0.955, "", ha="center", va="top", fontsize=9, family="monospace")
    fig.subplots_adjust(top=0.90)
    last_status_t = -STATUS_INTERVAL_S  # sofort beim ersten Frame anzeigen

    def update(_frame):
        nonlocal last_status_t
        q, _, _ = rebotarm.get_state(request_feedback=False)  # kein Extra-Buszugriff -- stoert sonst gravity_hold() (500Hz)
        pos, rpy = joint_to_pose(q)
        rpy_deg = np.degrees(rpy)
        gripper_vec = pin.rpy.rpyToMatrix(*rpy) @ np.array([1.0, 0.0, 0.0])  # Greifrichtung = lokale X-Achse von end_link (per FK-Test verifiziert, nicht Z)
        with _state_lock:
            _latest_state["q"] = q.copy()
            _latest_state["pos"] = pos
            _latest_state["rpy"] = rpy
        t_buf.append(time.time() - t0)
        for i, v in enumerate(list(pos) + [rpy_deg[0]]):
            val_buf[i].append(v)
        for i, v in enumerate(gripper_vec):
            vector_buf[i].append(v)
        viz.update(pad_q_for_model(viz.model, q))
        if t_buf[-1] - last_status_t >= STATUS_INTERVAL_S:
            last_status_t = t_buf[-1]
            line_text = (
                f"t={t_buf[-1]:6.1f}s  x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}  "
                f"roll={rpy_deg[0]:+6.1f}  "
                f"vec=({gripper_vec[0]:+.3f}, {gripper_vec[1]:+.3f}, {gripper_vec[2]:+.3f})"
            )
            status_text.set_text(line_text)
            with _status_lock:
                _status_holder["text"] = line_text
        for i, line in enumerate(lines):
            line.set_data(t_buf, val_buf[i])
            axes[i].relim()
            axes[i].autoscale_view()
        for i, line in enumerate(vector_lines):
            line.set_data(t_buf, vector_buf[i])
        if t_buf:
            for ax in axes:
                ax.set_xlim(max(0.0, t_buf[-1] - WINDOW_S), t_buf[-1] + 0.5)
        return lines + vector_lines + [status_text]

    ani = animation.FuncAnimation(
        fig, update, interval=int(1000 / SAMPLE_HZ), cache_frame_data=False,
    )
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nbeende...")
        rebotarm.disconnect()

    with _state_lock:
        return list(_session_saves)


def main() -> None:
    saved = run()
    print(f"{len(saved)} Pose(n) in dieser Sitzung gespeichert.")


if __name__ == "__main__":
    main()
