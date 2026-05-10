from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("face_landmarker.task")
FEATURE_LANDMARKS = {
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 98, 327,
    33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14,
    78, 308, 54, 284, 58, 288, 172, 397, 234, 454, 152,
}


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        "Missing face_landmarker.task.\n"
        "Download with:\n"
        "curl -L "
        "\"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task\" "
        "-o face_landmarker.task"
    )


def pixelate(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    sw = max(1, int(w * scale))
    sh = max(1, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def posterize(img: np.ndarray, levels: int) -> np.ndarray:
    factor = max(1, 256 // max(2, levels))
    return (img // factor) * factor


def quantize_bgr(color: np.ndarray, levels: int = 6) -> np.ndarray:
    factor = max(1, 256 // max(2, levels))
    return ((color // factor) * factor).astype(np.uint8)


def reduce_vertices(vertices: list[tuple[int, int, int, float]], target_count: int) -> list[tuple[int, int, int, float]]:
    if len(vertices) <= target_count:
        return vertices

    keep = [v for v in vertices if v[0] in FEATURE_LANDMARKS]
    slots_left = max(0, target_count - len(keep))
    if slots_left == 0:
        return keep[:target_count]

    other = [v for v in vertices if v[0] not in FEATURE_LANDMARKS]
    step = max(1, len(other) // slots_left)
    sampled = [other[i] for i in range(0, len(other), step)][:slots_left]

    dedup: dict[tuple[int, int], tuple[int, int, int, float]] = {}
    for item in keep + sampled:
        key = (item[1], item[2])
        if key not in dedup or item[0] in FEATURE_LANDMARKS:
            dedup[key] = item
    return list(dedup.values())[:target_count]


def triangle_average_color(img: np.ndarray, tri_pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri_pts, 255)
    mean = cv2.mean(img, mask=mask)[:3]
    return np.array(mean, dtype=np.uint8)


def gman_grade(base_color: np.ndarray, shade: float) -> np.ndarray:
    c = base_color.astype(np.float32)
    gray = np.dot(c, [0.114, 0.587, 0.299])
    target = np.array([110.0, 145.0, 130.0], dtype=np.float32)
    mixed = gray * 0.45 + target * 0.55
    lit = mixed * (0.42 + 0.85 * shade)
    return np.clip(lit, 0, 255).astype(np.uint8)


def triangle_shade(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return 0.55

    n = n / norm
    light = np.array([-0.35, -0.2, 0.92], dtype=np.float32)
    light = light / np.linalg.norm(light)
    d = float(np.dot(n, light))
    bands = [0.18, 0.34, 0.5, 0.66, 0.82]
    return min(bands, key=lambda b: abs(b - max(0.0, d)))


def blend_feature_detail(stylized_face: np.ndarray, original_img: np.ndarray, face_mask: np.ndarray, amount: float) -> np.ndarray:
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    detail = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    detail = np.clip((detail * 0.35) + 128.0, 0, 255).astype(np.uint8)
    detail_bgr = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)

    mixed = cv2.addWeighted(stylized_face, 1.0, detail_bgr, amount, -128.0 * amount)
    out = stylized_face.copy()
    out[face_mask == 255] = mixed[face_mask == 255]
    return out


def build_vertices(landmarks: list, w: int, h: int) -> list[tuple[int, int, int, float]]:
    verts = []
    for i, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        verts.append((i, x, y, float(lm.z)))
    return verts


def draw_low_poly_face(
    frame: np.ndarray,
    base_img: np.ndarray,
    landmarks: list,
    poly_count: int,
    face_levels: int,
    detail_preserve: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    vertices = build_vertices(landmarks, w, h)
    reduced_vertices = reduce_vertices(vertices, target_count=poly_count)
    points_2d = [(x, y) for _, x, y, _ in reduced_vertices]
    if len(points_2d) < 3:
        return frame

    hull = cv2.convexHull(np.array(points_2d, dtype=np.int32))
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    z_lookup = {(x, y): z for _, x, y, z in reduced_vertices}
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in set(points_2d):
        subdiv.insert(p)

    poly_face = frame.copy()
    triangles = subdiv.getTriangleList()
    for tri in triangles:
        tri_pts = np.array(
            [[int(tri[0]), int(tri[1])], [int(tri[2]), int(tri[3])], [int(tri[4]), int(tri[5])]],
            dtype=np.int32,
        )

        if (
            np.any(tri_pts[:, 0] < 0)
            or np.any(tri_pts[:, 0] >= w)
            or np.any(tri_pts[:, 1] < 0)
            or np.any(tri_pts[:, 1] >= h)
        ):
            continue

        cx, cy = int(np.mean(tri_pts[:, 0])), int(np.mean(tri_pts[:, 1]))
        if face_mask[cy, cx] == 0:
            continue

        p0 = (tri_pts[0][0], tri_pts[0][1])
        p1 = (tri_pts[1][0], tri_pts[1][1])
        p2 = (tri_pts[2][0], tri_pts[2][1])
        z0 = z_lookup.get(p0, 0.0)
        z1 = z_lookup.get(p1, 0.0)
        z2 = z_lookup.get(p2, 0.0)

        depth_scale = w * 0.6
        v0 = np.array([float(p0[0]), float(p0[1]), z0 * depth_scale], dtype=np.float32)
        v1 = np.array([float(p1[0]), float(p1[1]), z1 * depth_scale], dtype=np.float32)
        v2 = np.array([float(p2[0]), float(p2[1]), z2 * depth_scale], dtype=np.float32)
        shade = triangle_shade(v0, v1, v2)

        color = triangle_average_color(base_img, tri_pts)
        color = gman_grade(color, shade)
        color = quantize_bgr(color, levels=face_levels)

        cv2.fillConvexPoly(poly_face, tri_pts, color.tolist())
        cv2.polylines(poly_face, [tri_pts], True, (14, 16, 14), 1, lineType=cv2.LINE_8)

    poly_face = blend_feature_detail(poly_face, base_img, face_mask, amount=detail_preserve)
    frame[face_mask == 255] = poly_face[face_mask == 255]
    return frame


def draw_eye_bar(frame: np.ndarray, landmarks: list, bar_scale: float = 2.0, bar_thickness_ratio: float = 0.28) -> np.ndarray:
    h, w = frame.shape[:2]

    # MediaPipe eye corner anchors
    left_outer = landmarks[33]
    left_inner = landmarks[133]
    right_inner = landmarks[362]
    right_outer = landmarks[263]

    p_lo = np.array([left_outer.x * w, left_outer.y * h], dtype=np.float32)
    p_li = np.array([left_inner.x * w, left_inner.y * h], dtype=np.float32)
    p_ri = np.array([right_inner.x * w, right_inner.y * h], dtype=np.float32)
    p_ro = np.array([right_outer.x * w, right_outer.y * h], dtype=np.float32)

    left_eye = (p_lo + p_li) * 0.5
    right_eye = (p_ri + p_ro) * 0.5
    eye_mid = (left_eye + right_eye) * 0.5

    eye_vec = right_eye - left_eye
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1.0:
        return frame

    angle = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    bar_w = eye_dist * bar_scale
    bar_h = max(10.0, eye_dist * bar_thickness_ratio)

    rect = (tuple(eye_mid.tolist()), (bar_w, bar_h), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(frame, box, (0, 0, 0))
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jail filter: live webcam with retro pixel look and black eye bar(s)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--max-faces", type=int, default=3, help="Maximum faces to detect")
    parser.add_argument("--pixel-scale", type=float, default=0.24)
    parser.add_argument("--poster-levels", type=int, default=7)
    parser.add_argument("--bar-scale", type=float, default=2.0, help="Eye bar width multiplier")
    parser.add_argument("--bar-thickness", type=float, default=0.28, help="Eye bar thickness ratio")
    parser.add_argument("--contrast-alpha", type=float, default=1.14)
    parser.add_argument("--contrast-beta", type=float, default=-14.0)
    parser.add_argument("--track-every-n", type=int, default=2, help="Run detector every N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_faces = max(1, min(5, args.max_faces))
    model_path = ensure_model(MODEL_PATH)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=args.max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0
    last_landmarks = []
    last_ts = time.time()
    fps = 0.0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror mode
            output = pixelate(frame, scale=args.pixel_scale)
            output = posterize(output, levels=args.poster_levels)

            if frame_count % max(1, args.track_every_n) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)
                if results.face_landmarks:
                    last_landmarks = list(results.face_landmarks[: args.max_faces])
                else:
                    last_landmarks = []

            if last_landmarks:
                for face_landmarks in last_landmarks:
                    output = draw_eye_bar(
                        output,
                        face_landmarks,
                        bar_scale=args.bar_scale,
                        bar_thickness_ratio=args.bar_thickness,
                    )

            output = cv2.convertScaleAbs(output, alpha=args.contrast_alpha, beta=args.contrast_beta)

            now = time.time()
            dt = max(1e-6, now - last_ts)
            instant = 1.0 / dt
            fps = fps * 0.9 + instant * 0.1
            last_ts = now

            cv2.putText(
                output,
                f"FPS {fps:.1f} | Faces {len(last_landmarks)} | BarW {args.bar_scale:.2f} | BarH {args.bar_thickness:.2f}",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 250, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "Q/E bar width -/+ | A/D bar height -/+ | Z/C pixel -/+ | Esc quit",
                (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Jail filter", output)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                break
            if key == ord("q"):
                args.bar_scale = max(1.2, args.bar_scale - 0.08)
            elif key == ord("e"):
                args.bar_scale = min(3.5, args.bar_scale + 0.08)
            elif key == ord("a"):
                args.bar_thickness = max(0.12, args.bar_thickness - 0.02)
            elif key == ord("d"):
                args.bar_thickness = min(0.8, args.bar_thickness + 0.02)
            elif key == ord("z"):
                args.pixel_scale = max(0.14, args.pixel_scale - 0.02)
            elif key == ord("c"):
                args.pixel_scale = min(0.45, args.pixel_scale + 0.02)

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
