# Jail filter + Unity tracking service

This folder contains two tools:

1. **`jail_filter.py`** — Live mirror: pixel/poster look + black eye bar(s) on detected face(s).
2. **`tracker_service.py`** — Sends face + compact body pose over **UDP** as JSON (for a Unity client or any listener).

## Requirements

- Python **3.10–3.13** (if `mediapipe` fails on your version, use 3.11 or 3.12).
- Webcam.
- Model files next to the scripts (or download as below).

### Python packages

```bash
pip install -r requirements.txt
```

### Model files

| File | Used by | If missing |
|------|---------|------------|
| `face_landmarker.task` | `jail_filter.py` + `tracker_service.py` | Required. Download: [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
| `pose_landmarker_full.task` | `tracker_service.py` only (body) | Optional for face-only; body tracking tries to auto-download or use `curl` (see script). [pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task) |

Example download:

```bash
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o face_landmarker.task
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -o pose_landmarker_full.task
```

---

## Run: live webcam (eye bar + stylize)

```bash
python3 jail_filter.py
```

Useful options:

```bash
python3 jail_filter.py --camera-index 0 --max-faces 3 --bar-scale 2.0 --bar-thickness 0.28
```

**Keys while running:** `Q/E` bar width, `A/D` bar thickness, `Z/C` pixel scale, `Esc` quit.

---

## Run: UDP tracker service (Unity / debug)

Sends JSON packets to `host:port` (default `127.0.0.1:5053`).

```bash
python3 tracker_service.py --preview --log-sends
```

Common options:

```bash
python3 tracker_service.py --host 127.0.0.1 --port 5053 --stream-fps 20 --detect-every-n 2 --max-faces 3
```

- **`--preview`** — OpenCV window with face + skeleton overlay.
- **`--log-sends`** — Print send rate and face/pose counts each second.
- **`--disable-pose`** — Face only (no body model).
- **`--send-full-pose`** — Full pose landmarks (much larger UDP packets; usually avoid for Unity).

### Test UDP without Unity

Terminal A:

```bash
python3 tracker_service.py --log-sends
```

Terminal B:

```bash
python3 - <<'PY'
import json, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("127.0.0.1", 5053))
while True:
    data, _ = s.recvfrom(65535)
    msg = json.loads(data.decode("utf-8"))
    print("seq", msg.get("seq"), "faces", len(msg.get("faces", [])), "poses", len(msg.get("poses", [])))
PY
```

### Packet shape (short)

Each datagram is UTF-8 JSON: `type`, `version`, `timestamp_ms`, `seq`, `camera`, `faces` (each with `bbox` + downsampled `landmarks`), `poses` (compact joint lists by default).

---

## macOS notes

- If Python HTTPS downloads fail for models, use the `curl` commands above.
- Avoid suspending the process with `Ctrl+Z`; use `Ctrl+C` or close the preview window / `Esc` where supported.

---

## Mirror parity workflow (stick rig vs avatar)

For booth runtime (best stability), run tracker without preview:

```bash
python3 tracker_service.py --host 127.0.0.1 --port 5054 --stream-fps 20 --detect-every-n 2 --max-faces 1 --log-sends
```

In Unity:

- Set `TrackingReceiver.listenPort` to the same port (`5054` in the example).
- Add `TrackingDebugHUD` and assign:
  - `receiver`
  - `stickFigure` (`StickFigureDriver`)
  - `avatar` (`AvatarPoseDriver`)
- Keep both stick and avatar active while tuning.

### Fixed parity test routine

1. **Front-center hold (3s):** check mirror direction and baseline scale match.
2. **Left/right sweep:** both rigs should move same direction and similar distance.
3. **Step closer / step back:** `scale` and `z` trends should match between rigs.
4. **Edge crop test:** move partly out of frame; limbs should stop/vanish consistently.
5. **Re-enter center:** both rigs should recover to similar pose and scale quickly.

Use HUD lines:

- `Python mirror: scale=... z=...`
- `Stick output: scale=... z=...`
- `Avatar output: scale=... z=... root=(...)`

Tune until stick and avatar output values track each other closely under the same movement.
