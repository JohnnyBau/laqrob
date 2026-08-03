#!/usr/bin/env python3
"""Kombiniert Pick- und Lackier-Posen zu einer vollstaendigen Ablaufsequenz.

Fuer jedes Teil n aus poses_<partname>_pick.json (Schema <prefix>_pre_n/_rdy_n/
_open_n/_post_n) wird das Teil gegriffen und anschliessend die komplette
Lackiersequenz aus poses_<partname>_paint.json abgefahren.

Ablauf pro Teil n: <prefix>_trans_n, <prefix>_pre_n, <prefix>_rdy_n,
<prefix>_open_n, <prefix>_post_n, <lackierpose_1>_n, ...

Geschwindigkeit:
    Jede Quelldatei kann ein 'speed'-Feld (rad/s) enthalten (Wrapper-Format).
    Die CLI-Argumente --pick-speed / --paint-speed ueberschreiben den JSON-Wert.
    Jede Pose in der Ausgabe erhaelt ein 'speed'-Feld, das play.py fuer die
    Dauern-Berechnung verwendet.

Nutzung:
    uv run python example/poses_pick_paint_drop.py body1
    uv run python example/poses_pick_paint_drop.py body1 --pick-speed 0.3 --paint-speed 1.0
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

POSES_DIR = Path(__file__).resolve().parent

# Roboter-Nullstellung als sicherer Übergangspunkt vor jedem Greifen
START_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def load_poses(path: Path) -> tuple[list, float | None]:
    """Laedt Posen-JSON; unterstuetzt Array- und Wrapper-Format {'speed':..,'poses':[..]}.
    Gibt (poses, speed) zurueck; speed ist None wenn nicht angegeben."""
    if not path.exists():
        raise FileNotFoundError(f"Posendatei nicht gefunden: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, None
    return data["poses"], data.get("speed")


def save_poses(poses: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poses, indent=2, ensure_ascii=False))


def group_pick_poses(pick_poses: list, prefix: str) -> dict:
    """Gruppiert Pick-Posen nach Teilnummer n -> {step: pose}."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(pre|rdy|open|post)_(\d+)$")
    groups = defaultdict(dict)
    for pose in pick_poses:
        match = pattern.match(pose["name"])
        if match:
            step, n = match.group(1), int(match.group(2))
            groups[n][step] = pose
    return groups


def build_pick_paint_sequence(
    pick_poses: list, pick_speed: float | None,
    paint_poses: list, paint_speed: float | None,
    prefix: str,
) -> tuple[list, int]:
    groups = group_pick_poses(pick_poses, prefix)
    if not groups:
        raise ValueError(f"Keine Pick-Posen mit Praefix '{prefix}' gefunden.")
    sequence = []
    for n in sorted(groups):
        group = groups[n]
        trans = {"name": f"{prefix}_trans_{n}", "q": START_Q}
        if pick_speed is not None:
            trans["speed"] = pick_speed
        sequence.append(trans)
        for step in ("pre", "rdy", "open", "post"):
            if step not in group:
                raise ValueError(f"Teil {n}: Schritt '{prefix}_{step}_{n}' fehlt in den Pick-Posen.")
            pose = dict(group[step])
            if pick_speed is not None:
                pose["speed"] = pick_speed
            sequence.append(pose)
        for paint_pose in paint_poses:
            pose = dict(paint_pose)
            pose["name"] = f"{paint_pose['name']}_{n}"
            if paint_speed is not None:
                pose["speed"] = paint_speed
            sequence.append(pose)
    return sequence, len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("partname", help="Teilname, bestimmt Ein- und Ausgabedateien "
                                          "(poses_<partname>_pick.json / _paint.json / _pick_paint_drop.json)")
    parser.add_argument("--prefix", default="pick", help="Namenspraefix der Pick-Posen (Standard 'pick')")
    parser.add_argument("--pick-poses", type=Path, default=None,
                         help="Pick-Posen-Datei (Standard: poses_<partname>_pick.json)")
    parser.add_argument("--paint-poses", type=Path, default=None,
                         help="Lackier-Posen-Datei (Standard: poses_<partname>_paint.json)")
    parser.add_argument("--poses", type=Path, default=None,
                         help="Ziel-JSON-Datei (Standard: poses_<partname>_pick_paint_drop.json)")
    parser.add_argument("--pick-speed", type=float, default=None,
                         help="Pick-Geschwindigkeit [rad/s] (ueberschreibt JSON-Wert)")
    parser.add_argument("--paint-speed", type=float, default=None,
                         help="Lackier-Geschwindigkeit [rad/s] (ueberschreibt JSON-Wert)")
    args = parser.parse_args()

    pick_path = args.pick_poses if args.pick_poses is not None else POSES_DIR / f"poses_{args.partname}_pick.json"
    paint_path = args.paint_poses if args.paint_poses is not None else POSES_DIR / f"poses_{args.partname}_paint.json"
    out_path = args.poses if args.poses is not None else POSES_DIR / f"poses_{args.partname}_pick_paint_drop.json"

    print(f"Lade Pick-Posen aus {pick_path}")
    pick_poses, pick_speed_json = load_poses(pick_path)
    print(f"Lade Lackier-Posen aus {paint_path}")
    paint_poses, paint_speed_json = load_poses(paint_path)

    pick_speed = args.pick_speed if args.pick_speed is not None else pick_speed_json
    paint_speed = args.paint_speed if args.paint_speed is not None else paint_speed_json
    if pick_speed is not None:
        print(f"  Pick-Geschwindigkeit:  {pick_speed} rad/s")
    if paint_speed is not None:
        print(f"  Paint-Geschwindigkeit: {paint_speed} rad/s")

    sequence, n_parts = build_pick_paint_sequence(
        pick_poses, pick_speed, paint_poses, paint_speed, args.prefix
    )

    save_poses(sequence, out_path)
    print(f"-> {len(sequence)} Posen geschrieben ({n_parts} Teile x (1 Trans + 4 Pick + {len(paint_poses)} Lackier)) "
          f"in {out_path}")


if __name__ == "__main__":
    main()
