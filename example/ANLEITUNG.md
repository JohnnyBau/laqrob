# Anleitung: Neues Teil einlernen, simulieren und am Roboter ausführen

## Überblick

```
Schritt 1  poses_pick.py          → poses_<teil>_pick.json           (Pick-Raster)
Schritt 2  (manuell)              → poses_<teil>_paint.json          (Lackierposen)
Schritt 3  poses_pick_paint_drop.py → poses_<teil>_pick_paint_drop.json (Gesamtablauf)
Schritt 4  poses_traj_sim.py      → MeshCat                          (visuell prüfen)
Schritt 5  compile_q.py           → poses_<teil>_pick_paint_drop_q.json (IK → q)
Schritt 6  play_sim.py            → MeshCat                          (q-Daten prüfen, optional)
Schritt 7  play.py  (auf Pi)      → echter Roboter
```

---

## WSL-Befehle ausführen

Alle Simulations-Skripte laufen nur in WSL (Pinocchio läuft nicht unter Windows).

```powershell
# Einzelbefehl aus PowerShell:
& "C:\Windows\system32\wsl.exe" -e bash -lc "cd ~/reBotArm_control_py && <befehl>"

# Oder interaktive WSL-Shell öffnen:
wsl
cd ~/reBotArm_control_py
```

Nach Codeänderungen in VS Code alles nach WSL synchronisieren:
```powershell
& "C:\Windows\system32\wsl.exe" -e bash -lc "rsync -av --include='*/' --include='*.py' --include='*.json' --include='*.yaml' --exclude='*' /mnt/c/Users/z003n57f/Documents/000_B601/reBotArm_control_py/ ~/reBotArm_control_py/"
```

---

## Schritt 1 – Pick-Posen generieren (`poses_pick.py`)

Referenzpose des ersten Teils (vorne links in der Box) anfahren, Position ablesen.

```bash
uv run python example/poses_pick.py body1 \
    0.2 -0.36 0.17  0.0 -0.701 -0.701 \
    --dive-depth 0.04 --nx 4 --ny 2 --dx 0.15 --dy 0.1 \
    --speed 0.4
```

| Parameter | Bedeutung |
|---|---|
| `body1` | Teilname → Ausgabe `poses_body1_pick.json` |
| `x y z` | Position Referenzteil [m] |
| `vx vy vz` | Greifer-Eintauchrichtung (normiert) |
| `--dive-depth` | Eintauchtiefe [m] |
| `--nx / --ny` | Anzahl Teile in X / Y |
| `--dx / --dy` | Teileabstand [m] |
| `--nx-dir / --ny-dir` | Rasterrichtung (`-1` = abnehmend, `1` = zunehmend) |
| `--speed` | Gelenkgeschwindigkeit [rad/s], wird in JSON gespeichert |

→ Ausgabe: `example/poses_body1_pick.json`

Erzeugt pro Teil 4 Posen: `pick_pre_n` → `pick_rdy_n` → `pick_open_n` → `pick_post_n`

---

## Schritt 2 – Lackierposen anlegen

Datei `example/poses_<partname>_paint.json` manuell anlegen:

```json
{
  "speed": 0.8,
  "poses": [
    { "name": "paint_trans", "q": [-1.58, -0.08, -0.87, -0.79, 0.03, 6.27] },
    { "name": "paint_upper", "x": 0.0, "y": 0.2, "z": 0.75, "vx":  0.701, "vy": 0, "vz": 0.701 },
    { "name": "paint_vert",  "x": 0.0, "y": 0.2, "z": 0.75, "vx":  0,     "vy": 0, "vz": 1    },
    { "name": "paint_lower", "x": 0.0, "y": 0.2, "z": 0.75, "vx": -0.701, "vy": 0, "vz": 0.701 }
  ]
}
```

**Regeln:**
- `"speed"` [rad/s]: Gelenkgeschwindigkeit für alle Lackierposen
- Erste Pose: immer Gelenkraum-Übergang (`"q"`, kein `x/y/z`)
- Kartesische Posen: `x/y/z` + Richtungsvektor `vx/vy/vz`; optionales `"j6"` [rad] dreht das Werkzeug um die Eintauchachse

Beispiel: `example/poses_body1_paint.json`

---

## Schritt 3 – Gesamtablauf kombinieren (`poses_pick_paint_drop.py`)

```bash
uv run python example/poses_pick_paint_drop.py body1
```

Liest `poses_body1_pick.json` + `poses_body1_paint.json`, fügt Übergangsposen ein und
schreibt die Geschwindigkeit aus dem jeweiligen Wrapper in jede Pose.

Geschwindigkeit für diesen Lauf überschreiben:
```bash
uv run python example/poses_pick_paint_drop.py body1 --pick-speed 0.3 --paint-speed 0.6
```

→ Ausgabe: `example/poses_body1_pick_paint_drop.json`

| Pose | Typ | Beschreibung |
|---|---|---|
| `pick_trans_n` | Gelenkraum | Übergangskonfiguration vor dem Greifen |
| `pick_pre_n` | Kartesisch | Annäherung über dem Teil |
| `pick_rdy_n` | Kartesisch | Greifposition |
| `pick_open_n` | Kartesisch | Greifer öffnen / schließen |
| `pick_post_n` | Kartesisch | Rückzug mit Teil |
| `paint_trans_n` | Gelenkraum | Übergang zur Lackierposition |
| `paint_..._n` | Kartesisch | Lackierschritte |

---

## Schritt 4 – Simulation mit IK/CLIK (`poses_traj_sim.py`)

Prüft ob IK für alle Posen lösbar ist und zeigt die Trajektorie in MeshCat.

```bash
uv run python -u example/sim/poses_traj_sim.py example/poses_body1_pick_paint_drop.json
```

MeshCat öffnen: `http://127.0.0.1:7000/static/`

Optionen:
```bash
--pause 0.5          # Pause zwischen Posen [s]
--names p1 p2        # nur bestimmte Posen
--prefix pick        # nur Posen mit Namens-Präfix
```

Beenden mit **Strg+C**.

---

## Schritt 5 – Gelenkwinkel kompilieren (`compile_q.py`)

Löst IK für alle Posen offline (kein MeshCat, läuft in Sekunden) und speichert
die Gelenkwinkel `q` direkt in der JSON:

```bash
uv run python example/sim/compile_q.py example/poses_body1_pick_paint_drop.json
```

→ Ausgabe: `example/poses_body1_pick_paint_drop_q.json`

Abweichender Ausgabepfad:
```bash
uv run python example/sim/compile_q.py example/poses_body1_pick_paint_drop.json \
    --out example/poses_body1_pi.json
```

> **Nach jeder Änderung an Pick- oder Lackierposen erneut ausführen.**

---

## Schritt 6 – Simulation der q-Daten (`play_sim.py`) *(optional)*

Spielt die kompilierte q-JSON in MeshCat ab — kein IK, exakt dieselbe Min-Jerk-Logik
wie `play.py` auf dem Pi. Letzter Schritt vor dem echten Roboter.

```bash
uv run python example/sim/play_sim.py example/poses_body1_pick_paint_drop_q.json
uv run python example/sim/play_sim.py example/poses_body1_pick_paint_drop_q.json --scale 2.0
uv run python example/sim/play_sim.py example/poses_body1_pick_paint_drop_q.json --names pick_pre_1 pick_rdy_1
```

---

## Schritt 7 – Am echten Roboter ausführen (`play.py`)

> **Voraussetzung:** `_q.json` aus Schritt 5 vorhanden und Simulation aus Schritt 6 ohne Auffälligkeiten.

Dateien auf den Pi übertragen, dann auf dem Pi:

```bash
# Erstlauf: sehr langsam, Arm beobachten
uv run python example/play.py --poses example/poses_body1_pick_paint_drop_q.json --scale 5.0

# Normallauf
uv run python example/play.py --poses example/poses_body1_pick_paint_drop_q.json --scale 1.0
```

| Parameter | Bedeutung |
|---|---|
| `--poses` | Pfad zur `_q.json` (Pflichtfeld) |
| `--scale` | Zeitdehnungsfaktor: `>1` = langsamer, `1.0` = Normaltempo (Standard: 3.0) |
| `--speed` | Fallback-Geschwindigkeit [rad/s] für Posen ohne `speed`-Feld (Standard: 0.5) |
| `--dt` | Regelkreis-Zeitschritt [s] (Standard: 0.02 = 50 Hz) |

Bewegungszeit pro Pose:
```
duration = max(0.5 s,  max(|q_soll − q_ist|) / speed)  ×  scale
```

Der Arm fährt nach Abschluss oder bei Fehler automatisch in die Safe-Home-Position.





