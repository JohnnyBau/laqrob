# Live-Pose-Plot & Simulation (Free-Drive)

`example/live_pose_plot.py` visualisiert die Endeffektor-Pose des reBotArm live
im Browser, waehrend der Arm im Free-Drive (Gravitationskompensation) von
Hand gefuehrt wird. Der Arm bewegt sich dabei NICHT von selbst -- er haengt
weich in `gravity_hold()` (wie bei `teach.py`) und laesst sich mit der Hand
fuehren.

## Starten

```bash
cd ~/reBotArm_control_py
uv run python example/live_pose_plot.py
```

Beenden mit `Strg+C` im Terminal. Danach `rebotarm.disconnect()` (automatisch
in einer `finally`-Klausel), der Arm wird stromlos/undurchsichtig.

## Was laeuft parallel

Ein einziger Prozess startet drei Ausgaben gleichzeitig, alle per LAN vom PC
aus im Browser erreichbar (`http://<pi-ip>:<port>`, Pi-IP z.B. `192.168.178.130`):

| Port | Inhalt | Aktualisierung |
|------|--------|-----------------|
| 8989 | Live-Plot (matplotlib, 5 Diagramme) | ~20x/s |
| 8990 | Kopierbarer Status-Text (reines HTML) | alle 10s (Browser-Refresh alle 2s) |
| 7000 | 3D-Simulation des Arms (MeshCat) | ~20x/s |

## Live-Plot (Port 8989)

5 Diagramme uebereinander, X-Achse = Zeit (letzte 20s), Y-Achsen skalieren
automatisch (dynamisch):

- **x, y, z [m]** -- Position der Greiferspitze im Basiskoordinatensystem
  des Roboters.
- **roll [deg]** -- Rotation um die Greifrichtung (siehe unten), aus der
  FK berechnet.
- **Greifer-Vektor (vx, vy, vz)** -- Einheitsvektor, der angibt, wohin der
  Greifer zeigt (siehe Interpretation unten). Skala ist fest auf [-1.1, 1.1],
  da es sich um einen Einheitsvektor handelt.

Oben im Plot steht zusaetzlich die gleiche Statuszeile wie unter Port 8990
(aktualisiert alle 10s).

## Kopierbarer Status-Text (Port 8990)

Reine HTML-Seite (`<pre>`-Text), damit man die aktuellen Werte markieren und
kopieren kann (im Plot selbst, Port 8989, ist der Text Teil eines Canvas und
nicht markierbar). Aktualisiert sich alle 10s, Seite laedt sich alle 2s neu.

Format: `t=<Zeit>s  x=<..>  y=<..>  z=<..>  roll=<..>  vec=(vx, vy, vz)`

## 3D-Simulation (Port 7000/static/)

MeshCat-Ansicht des kompletten Roboters (aus dem URDF), die live die
tatsaechlichen Gelenkwinkel abbildet -- gut geeignet, um die Armhaltung auch
ohne direkten Blick auf die Hardware zu pruefen (z.B. von einem anderen Raum
aus).

## Interpretation: Greifrichtung / Greifer-Vektor

Der Endeffektor-Frame (`gripper_tip`, siehe `config/rebotarm_dm.yaml`) zeigt
mit seiner **lokalen X-Achse** in Greifrichtung (nicht Z -- das wurde per
FK-Test verifiziert: nur die X-Achse bleibt beim Drehen von Joint 6 [Handgelenk-
Rotation] konstant, Y/Z drehen sich mit). Der "Greifer-Vektor" im Plot ist
genau diese X-Achse, ausgedrueckt im Basiskoordinatensystem -- er zeigt also
dorthin, wo der Greifer gerade hinzeigt/-greift, unabhaengig vom Roll-Winkel
des Handgelenks.

`gripper_tip` liegt 7cm entlang dieser Achse vor `end_link` (der eigentlichen
Linearfuehrung des Greifers) -- das entspricht dem aktuell montierten
(laengeren) Greifer. Alle Positions-/IK-Berechnungen im Repo (FK, IK,
`compile_poses.py`, dieser Live-Plot, MeshCat) beziehen sich auf diesen Punkt.

## Sicherheits-/Technik-Hinweise

- `gravity_hold()` haelt den Arm weich (aktuell `kp=5.0, kd=2.0`), man kann
  ihn mit spuerbarem, aber ueberwindbarem Widerstand von Hand fuehren.
- Live-Plot und Simulation lesen den Zustand mit `request_feedback=False`
  (gecachte Werte) -- ein aktiver Bus-Poll wuerde mit der 500Hz-Regelschleife
  kollidieren und die Stabilitaet verschlechtern.
- Playback (`playback.py`) ist von alledem unberuehrt -- das laeuft rein im
  Gelenkwinkelraum (`move_to_q_traj`), nicht ueber die hier gezeigte Pose.
