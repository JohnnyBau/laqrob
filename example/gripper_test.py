#!/usr/bin/env python3
"""Testlauf: Greifer oeffnet (oder schliesst) mit konstanter Kraft (Kraftregelung,
kp=0), bis Widerstand erkannt wird (Gelenk steht trotz anliegendem Drehmoment still)
oder der definierte Verfahrweg --max-travel erreicht ist.

Zwei moegliche, beide normale (nicht-fehlerhafte) Endzustaende:
  - Widerstand VOR --max-travel erkannt  -> vermutlich Objekt gegriffen.
  - --max-travel ohne Widerstand erreicht -> vermutlich kein Objekt vorhanden,
    Meldung wird ausgegeben, Skript laeuft normal zu Ende (kein Fehler/Abbruch).

Nutzung:
    uv run python example/gripper_test.py                 # oeffnen, tau=+1.0 Nm
    uv run python example/gripper_test.py --tau -1.0       # andere Richtung testen
    uv run python example/gripper_test.py --tau 1.0 --max-travel 1.0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from reBotArm_control_py.actuator import RebotArm

from _common import (
    GRIPPER_KD_FORCE, GRIPPER_MAX_TRAVEL, GRIPPER_TIMEOUT, gripper_wait_for_event,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=1.0,
                        help="konstantes Drehmoment [Nm], Vorzeichen = Richtung (DM4310 Tmax=10)")
    parser.add_argument("--max-travel", type=float, default=GRIPPER_MAX_TRAVEL,
                        help="erwarteter freier Verfahrweg ohne Objekt [rad], ab Startposition "
                             "(Sicherheitsnetz gegen mechanischen Anschlag/Ueberdrehen)")
    parser.add_argument("--timeout", type=float, default=GRIPPER_TIMEOUT, help="Sicherheits-Timeout [s]")
    args = parser.parse_args()

    rebotarm = RebotArm()
    rebotarm.connect()
    try:
        if not rebotarm.has_gripper:
            print("Kein Greifer in der Konfiguration gefunden.")
            return

        rebotarm.gripper.mode_mit()
        rebotarm.gripper.enable()

        print(f"Greifer-Kraftregelung: tau={args.tau:+.2f} Nm, max-travel={args.max_travel} rad, "
              f"Timeout={args.timeout}s")

        event, g_pos, travel = gripper_wait_for_event(
            rebotarm, send=True, tau=args.tau,
            max_travel=args.max_travel, timeout=args.timeout,
        )

        if event == "resistance":
            print(f"Widerstand erkannt bei pos={g_pos:+.4f} rad (travel={travel:.3f} rad).")
        elif event == "max_travel":
            print(f"Kein Widerstand bis max-travel={args.max_travel} rad gefunden "
                  f"(pos={g_pos:+.4f} rad) -- vermutlich kein Objekt vorhanden. "
                  f"Normales Ende, kein Fehler.")
        else:
            print(f"Timeout erreicht (weder Widerstand noch max-travel) bei pos={g_pos:+.4f} rad "
                  f"-- pruefe Verkabelung/Kraftrichtung.")
    finally:
        # Kraft abschalten, bevor die Verbindung getrennt wird.
        if rebotarm.has_gripper:
            rebotarm.gripper.send_mit(
                pos=rebotarm.gripper.get_positions(request_feedback=False),
                vel=np.zeros(1), kp=np.zeros(1), kd=np.full(1, GRIPPER_KD_FORCE),
                tau=np.zeros(1),
            )
            time.sleep(0.1)
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
