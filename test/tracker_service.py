from __future__ import annotations

import argparse
import json
import os
import math
import socket
import subprocess
import time
import traceback
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


FACE_MODEL_PATH = Path("face_landmarker.task")
POSE_MODEL_PATH = Path("pose_landmarker_full.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
MAX_UDP_BYTES = 60_000
SAFE_UDP_BYTES = 8_000
POSE_KEYPOINTS = [
    0,   # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
]
POSE_EDGES = [
    (0, 1), (0, 2),  # nose to shoulders
    (1, 2),          # shoulder line
    (1, 3), (3, 5),  # left arm
    (2, 4), (4, 6),  # right arm
    (1, 7), (2, 8),  # shoulders to hips
    (7, 8),          # hip line
    (7, 9), (9, 11), # left leg
    (8, 10), (10, 12),  # right leg
]

# BlazePose indices for elbow pole triangles (shoulder, elbow, wrist).
_LEFT_ARM_CHAIN = (11, 13, 15)
_RIGHT_ARM_CHAIN = (12, 14, 16)
_ANKLE_LEFT, _ANKLE_RIGHT = 27, 28
_NUM_POSE_LANDMARKS = 33


def _safe_float(v: object, default: float = 0.0) -> float:
    """MediaPipe sometimes leaves coordinates or scores unset (None); avoid TypeError / NaN propagation."""
    try:
        if v is None:
            return default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError, OverflowError):
        return default


def _sanitize_for_json(obj: object) -> object:
    """Replace NaN/Inf so json.dumps(..., allow_nan=False) is safe for strict Unity parsers."""
    if isinstance(obj, float):
        return 0.0 if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _safe_round(x: object, ndigits: int, default: float = 0.0) -> float:
    """Avoid OverflowError from round(inf, n) on bad landmark math."""
    v = _safe_float(x, default)
    try:
        return float(round(v, ndigits))
    except (OverflowError, ValueError):
        return default


def _euro_alpha(cutoff: float, te: float) -> float:
    te = max(te, 1e-9)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter1D:
    """1D One Euro Filter (Casiez et al.) for jitter reduction with low lag on motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def filter(self, x: float, te: float) -> float:
        prev = self._x_prev
        fallback = prev if (prev is not None and math.isfinite(prev)) else 0.0
        x = _safe_float(x, fallback)
        te = max(te, 1e-9)
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) / te
        a_d = _euro_alpha(self.d_cutoff, te)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        if not math.isfinite(dx_hat):
            dx_hat = 0.0
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _euro_alpha(cutoff, te)
        x_hat = a * x + (1.0 - a) * self._x_prev
        if not math.isfinite(x_hat):
            x_hat = x if math.isfinite(x) else fallback
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class LandmarkEuroBank:
    """One Euro filter per coordinate for a fixed-length flat landmark buffer."""

    def __init__(self, dim: int, min_cutoff: float, beta: float, d_cutoff: float):
        self.dim = dim
        self._filters = [OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(dim)]
        self._last_t: float | None = None

    def reset(self) -> None:
        self._last_t = None
        for f in self._filters:
            f.reset()

    def apply(self, values: list[float], t_now: float) -> list[float]:
        if len(values) != self.dim:
            raise ValueError(f"expected {self.dim} values, got {len(values)}")
        te = 0.033 if self._last_t is None else max(1e-4, t_now - self._last_t)
        self._last_t = t_now
        return [self._filters[i].filter(values[i], te) for i in range(self.dim)]


def must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def ensure_pose_model(path: Path) -> bool:
    if path.exists():
        return True

    print(f"[tracker_service] pose model missing, attempting download: {path}")
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
        return path.exists()
    except Exception:
        pass

    # Fallback for environments where Python SSL cert chain is not configured.
    try:
        result = subprocess.run(
            ["curl", "-L", POSE_MODEL_URL, "-o", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and path.exists():
            return True
    except Exception:
        pass
    return False


def face_bbox_norm(face_landmarks: list) -> dict[str, float]:
    if not face_landmarks:
        return {"cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.0}
    xs = [_safe_float(lm.x) for lm in face_landmarks]
    ys = [_safe_float(lm.y) for lm in face_landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "cx": (min_x + max_x) * 0.5,
        "cy": (min_y + max_y) * 0.5,
        "w": max_x - min_x,
        "h": max_y - min_y,
    }


def landmark_to_dict(lm) -> dict[str, float]:
    d: dict[str, float] = {
        "x": _safe_float(lm.x),
        "y": _safe_float(lm.y),
        "z": _safe_float(getattr(lm, "z", 0.0)),
    }
    # Short keys save UDP bytes. Unity uses v to hide guessed/occluded limbs (esp. legs).
    v = getattr(lm, "visibility", None)
    if v is not None:
        d["v"] = _safe_float(v, 1.0)
    pr = getattr(lm, "presence", None)
    if pr is not None:
        d["p"] = _safe_float(pr, 1.0)
    return d


def world_landmarks_to_flat(wl: list) -> list[float]:
    """33 * (x,y,z) in meters; pads if model returns fewer points."""
    out: list[float] = []
    for i in range(_NUM_POSE_LANDMARKS):
        if i < len(wl):
            lm = wl[i]
            out.extend(
                [
                    _safe_float(lm.x),
                    _safe_float(lm.y),
                    _safe_float(getattr(lm, "z", 0.0)),
                ]
            )
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


def flat_to_points33(flat: list[float]) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for i in range(0, _NUM_POSE_LANDMARKS * 3, 3):
        if i + 2 < len(flat):
            pts.append((flat[i], flat[i + 1], flat[i + 2]))
        else:
            pts.append((0.0, 0.0, 0.0))
    return pts


def triangle_unit_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    le = math.hypot(nx, ny, nz)
    if le < 1e-7:
        return None
    return (nx / le, ny / le, nz / le)


def elbow_poles_from_points33(
    pts: list[tuple[float, float, float]],
    prev: tuple[list[float], list[float]] | None,
) -> tuple[list[float], list[float]]:
    """Unit normals for (shoulder, elbow, wrist) arm triangles; pole-vector hint for IK."""

    def pole_for(chain: tuple[int, int, int]) -> list[float] | None:
        si, ei, wi = chain
        if si >= len(pts) or ei >= len(pts) or wi >= len(pts):
            return None
        n = triangle_unit_normal(pts[si], pts[ei], pts[wi])
        if n is None:
            return None
        return [float(n[0]), float(n[1]), float(n[2])]

    left = pole_for(_LEFT_ARM_CHAIN)
    right = pole_for(_RIGHT_ARM_CHAIN)
    if left is None and prev is not None:
        left = list(prev[0])
    if right is None and prev is not None:
        right = list(prev[1])
    if left is None:
        left = [0.0, 0.0, 0.0]
    if right is None:
        right = [0.0, 0.0, 0.0]
    return left, right


def compact_visibility(image_pose: list) -> list[float]:
    vis: list[float] = []
    for idx in POSE_KEYPOINTS:
        if idx < len(image_pose):
            lm = image_pose[idx]
            v = getattr(lm, "visibility", None)
            vis.append(_safe_float(v, 1.0) if v is not None else 1.0)
        else:
            vis.append(0.0)
    return vis


def mean_joint_visibility(image_pose: list) -> float:
    if not image_pose:
        return 0.0
    acc = 0.0
    n = 0
    for idx in POSE_KEYPOINTS:
        if idx < len(image_pose):
            v = getattr(image_pose[idx], "visibility", None)
            acc += _safe_float(v, 1.0) if v is not None else 1.0
            n += 1
    return acc / max(1, n)


def compact_world_xyz(smoothed_flat: list[float]) -> list[list[float]]:
    pts = flat_to_points33(smoothed_flat)
    out: list[list[float]] = []
    for idx in POSE_KEYPOINTS:
        if idx < len(pts):
            x, y, z = pts[idx]
            out.append([_safe_round(x, 5), _safe_round(y, 5), _safe_round(z, 5)])
        else:
            out.append([0.0, 0.0, 0.0])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam tracking service for Unity (UDP JSON)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5053)
    p.add_argument("--stream-fps", type=float, default=20.0)
    p.add_argument(
        "--detect-every-n",
        type=int,
        default=2,
        help="Run inference every N camera frames (higher = less CPU; tracking updates less often)",
    )
    p.add_argument("--max-faces", type=int, default=3, help="Maximum faces to send")
    p.add_argument("--max-face-landmarks", type=int, default=48, help="Downsampled face points sent to Unity")
    p.add_argument("--send-full-pose", action="store_true", help="Send all pose landmarks (larger packets)")
    p.add_argument("--disable-pose", action="store_true", help="Disable body pose tracking")
    p.add_argument("--preview", action="store_true", help="Show local debug preview window")
    p.add_argument(
        "--preview-downscale",
        type=int,
        default=640,
        help="Max preview width in pixels (smaller = less lag; 0 = full camera size)",
    )
    p.add_argument(
        "--preview-max-fps",
        type=float,
        default=30.0,
        help="Cap cv2.imshow rate (0 = unlimited). Lowers UI/GUI load on macOS.",
    )
    p.add_argument("--log-sends", action="store_true", help="Print send stats every second")
    p.add_argument(
        "--no-mirror-hint",
        action="store_true",
        help="Do not send mirror_scale / mirror_z_pull (Unity computes locally)",
    )
    p.add_argument("--mirror-ref-face-h", type=float, default=0.32)
    p.add_argument("--mirror-ref-shoulder", type=float, default=0.28)
    p.add_argument("--mirror-ref-body-h", type=float, default=0.70)
    p.add_argument("--mirror-blend", type=float, default=0.45)
    p.add_argument("--mirror-gain", type=float, default=1.0)
    p.add_argument("--mirror-min", type=float, default=0.45)
    p.add_argument("--mirror-max", type=float, default=2.6)
    p.add_argument(
        "--mirror-smooth",
        type=float,
        default=0.14,
        help="0..1 blend toward new target each sent packet (lower = calmer)",
    )
    p.add_argument(
        "--mirror-max-step",
        type=float,
        default=0.1,
        help="Max |delta| toward raw mirror target per packet (stops wild swings)",
    )
    p.add_argument("--mirror-z-gain", type=float, default=0.25)
    p.add_argument(
        "--disable-euro",
        action="store_true",
        help="Disable One Euro smoothing on world landmarks (raw pose_world_landmarks)",
    )
    p.add_argument("--euro-min-cutoff", type=float, default=1.0, help="One Euro min cutoff frequency (Hz)")
    p.add_argument("--euro-beta", type=float, default=0.007, help="One Euro speed coefficient")
    p.add_argument("--euro-d-cutoff", type=float, default=1.0, help="One Euro derivative cutoff (Hz)")
    p.add_argument(
        "--present-vis-threshold",
        type=float,
        default=0.35,
        help="Mean joint visibility above this marks is_present true",
    )
    p.add_argument(
        "--floor-window",
        type=int,
        default=45,
        help="Rolling sample count for floor_y from ankle world Y (MediaPipe Y grows toward feet)",
    )
    return p.parse_args()


def downsample_indices(count: int, target: int) -> list[int]:
    if target >= count:
        return list(range(count))
    step = max(1, count // target)
    out = list(range(0, count, step))[:target]
    return out


def encode_payload(payload: dict) -> bytes:
    clean = _sanitize_for_json(payload)
    return json.dumps(clean, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compact_pose(pose_landmarks: list) -> list[dict[str, float]]:
    compact = []
    for idx in POSE_KEYPOINTS:
        if idx < len(pose_landmarks):
            compact.append(landmark_to_dict(pose_landmarks[idx]))
    return compact


def shoulder_span_full_pose(pose_landmarks: list) -> float:
    if len(pose_landmarks) < 29:
        return 0.2
    p11, p12 = pose_landmarks[11], pose_landmarks[12]
    return math.hypot(
        _safe_float(p12.x) - _safe_float(p11.x),
        _safe_float(p12.y) - _safe_float(p11.y),
    )


def body_height_full_pose(pose_landmarks: list) -> float:
    if len(pose_landmarks) < 29:
        return 0.2
    nose_y = _safe_float(pose_landmarks[0].y)
    ankles_y = (_safe_float(pose_landmarks[27].y) + _safe_float(pose_landmarks[28].y)) * 0.5
    return abs(nose_y - ankles_y)


def compute_mirror_hints(
    face_landmarks_list: list,
    pose_landmarks_list: list,
    ref_face_h: float,
    ref_shoulder: float,
    ref_body_h: float,
    body_blend: float,
    gain: float,
    lo: float,
    hi: float,
    z_gain: float,
) -> tuple[float, float]:
    """Returns (target_scale, target_z_pull) before packet smoothing."""
    if not face_landmarks_list and not pose_landmarks_list:
        return 1.0, 0.0

    face_h = 0.0
    if face_landmarks_list:
        bb = face_bbox_norm(face_landmarks_list[0])
        face_h = float(bb["h"])

    pose = pose_landmarks_list[0] if pose_landmarks_list else None

    shoulder_w = shoulder_span_full_pose(pose) if pose else 0.0
    body_h = body_height_full_pose(pose) if pose else 0.0

    shoulder_w = max(shoulder_w, 0.06)
    body_h = max(body_h, 0.1)

    if face_landmarks_list and face_h > 0.05:
        face_h = max(face_h, 0.06)
        r_face = face_h / max(ref_face_h, 0.08)
    else:
        r_face = body_h / max(ref_body_h, 0.15) if pose else 1.0

    r_shoulder = shoulder_w / max(ref_shoulder, 0.08) if pose else r_face
    r_body = body_h / max(ref_body_h, 0.15) if pose else r_face

    ratio = (1.0 - body_blend) * r_face + body_blend * r_shoulder
    ratio = 0.75 * ratio + 0.25 * r_body

    target_scale = max(lo, min(hi, ratio * gain))
    z_pull = -(ratio - 1.0) * z_gain
    return target_scale, z_pull


def draw_face_preview(frame, face_landmarks: list, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    bbox = face_bbox_norm(face_landmarks)
    x1 = int((bbox["cx"] - bbox["w"] * 0.5) * w)
    y1 = int((bbox["cy"] - bbox["h"] * 0.5) * h)
    x2 = int((bbox["cx"] + bbox["w"] * 0.5) * w)
    y2 = int((bbox["cy"] + bbox["h"] * 0.5) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Sparse draw so preview stays fast.
    for i in range(0, len(face_landmarks), 12):
        lm = face_landmarks[i]
        x = int(_safe_float(lm.x) * w)
        y = int(_safe_float(lm.y) * h)
        cv2.circle(frame, (x, y), 2, color, -1)


def draw_pose_preview(
    frame,
    pose_landmarks: list,
    color: tuple[int, int, int],
    min_visibility: float = 0.35,
) -> None:
    h, w = frame.shape[:2]
    pts = []
    vis = []
    for lm in pose_landmarks:
        x = int(_safe_float(lm.x) * w)
        y = int(_safe_float(lm.y) * h)
        pts.append((x, y))
        vis.append(_safe_float(getattr(lm, "visibility", 1.0), 1.0))

    # LINE_8 is much cheaper than LINE_AA; preview stays responsive on Retina/macOS.
    for a, b in POSE_EDGES:
        if a < len(pts) and b < len(pts):
            if vis[a] >= min_visibility and vis[b] >= min_visibility:
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_8)

    for i, p in enumerate(pts):
        if vis[i] >= min_visibility:
            cv2.circle(frame, p, 3, color, -1, lineType=cv2.LINE_8)


def main() -> None:
    # Before heavy work: less MediaPipe/TFLite chatter on stderr (can steal time on some setups).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    args = parse_args()
    args.detect_every_n = max(1, args.detect_every_n)
    args.max_faces = max(1, min(3, args.max_faces))
    args.max_face_landmarks = max(8, min(478, args.max_face_landmarks))
    must_exist(FACE_MODEL_PATH, "face model")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    endpoint = (args.host, args.port)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    face_base = mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=args.max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_landmarker = None
    pose_enabled = not args.disable_pose
    if pose_enabled and ensure_pose_model(POSE_MODEL_PATH):
        pose_base = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=args.max_faces,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    elif pose_enabled:
        print("[tracker_service] pose model unavailable; continuing with face only.")

    target_dt = 1.0 / max(1.0, args.stream_fps)
    last_send = 0.0
    fps = 0.0
    last_tick = time.time()
    start_monotonic = time.monotonic()
    seq = 0
    frame_idx = 0
    last_faces_raw = []
    last_poses_raw = []
    last_poses_world_raw: list[list] = []
    sent_count = 0
    last_log_t = time.time()
    read_fail_count = 0
    mirror_scale_smoothed = 1.0
    mirror_z_smoothed = 0.0
    last_preview_imshow_t = 0.0

    world_flat_dim = _NUM_POSE_LANDMARKS * 3
    euro_banks = [
        LandmarkEuroBank(
            world_flat_dim,
            args.euro_min_cutoff,
            args.euro_beta,
            args.euro_d_cutoff,
        )
        for _ in range(args.max_faces)
    ]
    floor_samples: deque[float] = deque(maxlen=max(5, args.floor_window))
    last_elbow_poles: list[tuple[list[float], list[float]] | None] = [None] * args.max_faces

    print(
        f"[tracker_service] start cam={args.camera_index} {args.camera_width}x{args.camera_height} "
        f"stream_fps={args.stream_fps} detect_every_n={args.detect_every_n} "
        f"max_faces={args.max_faces} max_face_landmarks={args.max_face_landmarks} "
        f"udp={args.host}:{args.port}"
    )

    try:
        with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    read_fail_count += 1
                    if read_fail_count % 30 == 0:
                        print(f"[tracker_service] warning: camera read failed x{read_fail_count}")
                    time.sleep(0.005)
                    continue
                read_fail_count = 0

                frame = cv2.flip(frame, 1)  # mirror-mode tracking for booth UX

                now = time.time()
                run_detect = (frame_idx % max(1, args.detect_every_n) == 0)
                if run_detect:
                    # Only pay for BGR→RGB + Image wrapper on frames we actually infer.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    # Tasks VIDEO mode expects strictly increasing frame timestamps.
                    timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)
                    try:
                        face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms) if pose_landmarker is not None else None
                        last_faces_raw = face_results.face_landmarks[: args.max_faces] if face_results.face_landmarks else []
                        last_poses_raw = pose_results.pose_landmarks[: args.max_faces] if (pose_results is not None and pose_results.pose_landmarks) else []
                        if pose_results is not None and pose_results.pose_world_landmarks:
                            last_poses_world_raw = [list(pw) for pw in pose_results.pose_world_landmarks[: args.max_faces]]
                        else:
                            last_poses_world_raw = []
                    except Exception as exc:
                        print(f"[tracker_service] detect error: {exc}")
                        last_faces_raw = []
                        last_poses_raw = []
                        last_poses_world_raw = []

                if now - last_send >= target_dt:
                    try:
                        # Build JSON lists only when we send (~stream-fps), not every camera frame.
                        faces_payload = []
                        for face in last_faces_raw:
                            idx = downsample_indices(len(face), args.max_face_landmarks)
                            sampled = [landmark_to_dict(face[i]) for i in idx]
                            faces_payload.append(
                                {
                                    "bbox": face_bbox_norm(face),
                                    "landmarks": sampled,
                                    "landmark_count_source": len(face),
                                    "landmark_count_sent": len(sampled),
                                }
                            )

                        poses_payload = []
                        for pose in last_poses_raw:
                            if args.send_full_pose:
                                poses_payload.append([landmark_to_dict(p) for p in pose])
                            else:
                                poses_payload.append(compact_pose(pose))

                        world_poses_out: list[list[list[float]]] = []
                        pose_vis_out: list[list[float]] = []
                        elbow_out: list[list[list[float]]] = []
                        primary_visible = 0.0
                        primary_present = False

                        n_pose = len(last_poses_raw)
                        for slot in range(args.max_faces):
                            if slot >= n_pose:
                                euro_banks[slot].reset()
                                last_elbow_poles[slot] = None

                        for i in range(n_pose):
                            img_pose = last_poses_raw[i]
                            has_world = i < len(last_poses_world_raw) and bool(last_poses_world_raw[i])
                            if has_world:
                                wlm = last_poses_world_raw[i]
                                flat = world_landmarks_to_flat(wlm)
                                if args.disable_euro:
                                    sm_flat = flat
                                else:
                                    sm_flat = euro_banks[i].apply(flat, now)
                                pts = flat_to_points33(sm_flat)
                                wcompact = compact_world_xyz(sm_flat)
                                visc = compact_visibility(img_pose)
                                lv, rv = elbow_poles_from_points33(pts, last_elbow_poles[i])
                                last_elbow_poles[i] = (lv, rv)
                                elbow_out.append(
                                    [
                                        [_safe_round(lv[0], 4), _safe_round(lv[1], 4), _safe_round(lv[2], 4)],
                                        [_safe_round(rv[0], 4), _safe_round(rv[1], 4), _safe_round(rv[2], 4)],
                                    ]
                                )
                                world_poses_out.append(wcompact)
                                pose_vis_out.append([_safe_round(v, 4) for v in visc])
                                mv = mean_joint_visibility(img_pose)
                                if i == 0:
                                    primary_visible = mv
                                    primary_present = mv >= args.present_vis_threshold
                                if i == 0 and len(pts) > max(_ANKLE_LEFT, _ANKLE_RIGHT):
                                    ankle_y = max(pts[_ANKLE_LEFT][1], pts[_ANKLE_RIGHT][1])
                                    floor_samples.append(ankle_y)
                            else:
                                euro_banks[i].reset()
                                last_elbow_poles[i] = None
                                world_poses_out.append([[0.0, 0.0, 0.0] for _ in POSE_KEYPOINTS])
                                pose_vis_out.append([_safe_round(v, 4) for v in compact_visibility(img_pose)])
                                elbow_out.append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                                if i == 0:
                                    primary_visible = mean_joint_visibility(img_pose)
                                    primary_present = primary_visible >= args.present_vis_threshold

                        floor_y_val = sum(floor_samples) / len(floor_samples) if floor_samples else 0.0

                        payload = {
                            "type": "tracking_frame",
                            "version": 2 if pose_landmarker is not None else 1,
                            "timestamp_ms": int(now * 1000),
                            "seq": seq,
                            "camera": {"w": int(frame.shape[1]), "h": int(frame.shape[0])},
                            "faces": faces_payload,
                            "poses": poses_payload,
                        }
                        if pose_landmarker is not None:
                            payload["world_poses"] = world_poses_out
                            payload["pose_visibility"] = pose_vis_out
                            if pose_vis_out:
                                payload["visibility"] = pose_vis_out[0]
                            payload["elbow_poles"] = elbow_out
                            payload["is_visible"] = _safe_round(primary_visible, 4)
                            payload["is_present"] = primary_present
                            payload["floor_y"] = _safe_round(floor_y_val, 5)
                            payload["fps"] = _safe_round(fps, 2)
                        if not args.no_mirror_hint:
                            t_scale, t_z = compute_mirror_hints(
                                last_faces_raw,
                                last_poses_raw,
                                args.mirror_ref_face_h,
                                args.mirror_ref_shoulder,
                                args.mirror_ref_body_h,
                                max(0.0, min(1.0, args.mirror_blend)),
                                args.mirror_gain,
                                args.mirror_min,
                                args.mirror_max,
                                args.mirror_z_gain,
                            )
                            # Spike rejection vs last smoothed value (flickering face/pose)
                            if mirror_scale_smoothed > 0.1:
                                spike = t_scale / mirror_scale_smoothed
                                if spike > 1.42:
                                    t_scale = mirror_scale_smoothed * 1.42
                                elif spike < 0.72:
                                    t_scale = mirror_scale_smoothed * 0.72
                            # Hard cap per-packet move
                            step = max(0.02, args.mirror_max_step)
                            delta = t_scale - mirror_scale_smoothed
                            if delta > step:
                                t_scale = mirror_scale_smoothed + step
                            elif delta < -step:
                                t_scale = mirror_scale_smoothed - step

                            a = max(0.05, min(0.95, args.mirror_smooth))
                            mirror_scale_smoothed = mirror_scale_smoothed * (1.0 - a) + t_scale * a
                            mirror_z_smoothed = mirror_z_smoothed * (1.0 - a) + t_z * a
                            payload["mirror_scale"] = _safe_round(mirror_scale_smoothed, 4)
                            payload["mirror_z_pull"] = _safe_round(mirror_z_smoothed, 4)
                        encoded = encode_payload(payload)
                        if len(encoded) > SAFE_UDP_BYTES:
                            # 0) Drop duplicate joint visibility (Unity can use pose_visibility)
                            payload.pop("visibility", None)
                            encoded = encode_payload(payload)
                        if len(encoded) > SAFE_UDP_BYTES:
                            # 1) Drop face landmark density
                            for face in payload["faces"]:
                                lm = face["landmarks"]
                                if len(lm) > 16:
                                    face["landmarks"] = lm[::2]
                                    face["landmark_count_sent"] = len(face["landmarks"])
                            encoded = encode_payload(payload)
                        if len(encoded) > SAFE_UDP_BYTES:
                            # 2) Keep only first face
                            payload["faces"] = payload["faces"][:1]
                            payload["poses"] = payload["poses"][:1]
                            if pose_landmarker is not None:
                                wp = payload.get("world_poses")
                                if isinstance(wp, list) and wp:
                                    payload["world_poses"] = wp[:1]
                                pv = payload.get("pose_visibility")
                                if isinstance(pv, list) and pv:
                                    payload["pose_visibility"] = pv[:1]
                                    payload["visibility"] = payload["pose_visibility"][0]
                                ep = payload.get("elbow_poles")
                                if isinstance(ep, list) and ep:
                                    payload["elbow_poles"] = ep[:1]
                            encoded = encode_payload(payload)
                        if len(encoded) > SAFE_UDP_BYTES:
                            # 3) Drop face landmarks, keep bbox + pose
                            for face in payload["faces"]:
                                face["landmarks"] = []
                                face["landmark_count_sent"] = 0
                            encoded = encode_payload(payload)
                        if len(encoded) > SAFE_UDP_BYTES:
                            # 4) Never send empty poses (breaks Unity). Strip faces only and retry once.
                            payload["faces"] = []
                            encoded = encode_payload(payload)
                        if len(encoded) > SAFE_UDP_BYTES:
                            print(
                                f"[tracker_service] warning: packet still {len(encoded)} bytes; "
                                "skip send (reduce --send-full-pose or landmarks)"
                            )
                            encoded = None

                        if encoded is not None and len(encoded) <= SAFE_UDP_BYTES:
                            try:
                                sock.sendto(encoded, endpoint)
                                seq += 1
                                sent_count += 1
                                last_send = now
                            except OSError as exc:
                                print(f"[tracker_service] send error: {exc}")
                        elif encoded is not None:
                            print(f"[tracker_service] warning: packet too large ({len(encoded)} bytes), dropped")

                    except Exception as exc:
                        print(f"[tracker_service] send/build error: {exc}")
                        traceback.print_exc()
                        for _eb in euro_banks:
                            _eb.reset()
                        for _si in range(args.max_faces):
                            last_elbow_poles[_si] = None
                        last_send = now
                if args.log_sends and now - last_log_t >= 1.0:
                    print(
                        f"[tracker_service] fps={fps:.1f} sent={sent_count}/s "
                        f"faces={len(last_faces_raw)} poses={len(last_poses_raw)} "
                        f"to={args.host}:{args.port}"
                    )
                    sent_count = 0
                    last_log_t = now

                dt = max(1e-6, now - last_tick)
                fps = fps * 0.9 + (1.0 / dt) * 0.1
                last_tick = now

                if args.preview:
                    fh, fw = frame.shape[:2]
                    max_pw = max(0, int(args.preview_downscale))
                    if max_pw > 0 and fw > max_pw:
                        scale = max_pw / float(fw)
                        nh = max(1, int(round(fh * scale)))
                        preview = cv2.resize(frame, (max_pw, nh), interpolation=cv2.INTER_AREA)
                    else:
                        preview = frame

                    colors = [(80, 250, 80), (80, 180, 255), (230, 170, 80)]
                    for i, face in enumerate(last_faces_raw):
                        draw_face_preview(preview, face, colors[i % len(colors)])
                    for i, pose in enumerate(last_poses_raw):
                        compact_for_draw = (
                            pose
                            if args.send_full_pose
                            else [pose[idx] for idx in POSE_KEYPOINTS if idx < len(pose)]
                        )
                        draw_pose_preview(preview, compact_for_draw, colors[i % len(colors)])

                    cv2.putText(
                        preview,
                        f"Service FPS {fps:.1f}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_8,
                    )
                    cv2.putText(
                        preview,
                        f"Faces {len(last_faces_raw)} Poses {len(last_poses_raw)}",
                        (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (220, 220, 220),
                        1,
                        cv2.LINE_8,
                    )
                    cv2.putText(
                        preview,
                        f"Detect every {max(1, args.detect_every_n)} frame(s)",
                        (8, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_8,
                    )
                    cv2.putText(
                        preview,
                        "Esc quit",
                        (8, 84),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_8,
                    )

                    pmf = float(args.preview_max_fps)
                    if pmf <= 0.0 or (now - last_preview_imshow_t) >= (1.0 / pmf):
                        cv2.imshow("tracker_service preview", preview)
                        last_preview_imshow_t = now
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        if pose_landmarker is not None:
            pose_landmarker.close()


if __name__ == "__main__":
    main()
