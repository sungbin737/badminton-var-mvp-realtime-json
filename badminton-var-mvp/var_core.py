"""Geometry, event detection, and I/O for the no-fine-tuning badminton VAR MVP.

The pipeline intentionally returns UNKNOWN when the video evidence is weak.  It
is a decision-assistance prototype, not an official match officiating system.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Observation:
    time_ms: int
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class FusedObservation:
    time_ms: int
    court_x: float
    court_y: float
    confidence: float
    plane_disagreement_m: float
    a_image_xy: tuple[float, float]
    b_image_xy: tuple[float, float]


@dataclass(frozen=True)
class Event:
    type: str
    time_ms: int
    confidence: float
    details: dict[str, Any]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


class CameraCalibration:
    """Maps a point on the court floor between image pixels and court metres.

    The four input pixels must be ordered: near-left, near-right, far-right,
    far-left.  The corresponding court origin is the near-left outside corner.
    """

    def __init__(self, image_points: list[list[float]], width_m: float, length_m: float):
        if len(image_points) != 4:
            raise ValueError("court_image_points must contain exactly four points")
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(
            [[0.0, length_m], [width_m, length_m], [width_m, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        self.to_court = cv2.getPerspectiveTransform(src, dst)

    def image_to_court(self, x: float, y: float) -> tuple[float, float]:
        p = np.asarray([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, self.to_court)[0][0]
        return float(out[0]), float(out[1])


def _number(row: dict[str, str], names: Iterable[str], required: bool = True) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return float(value)
    if required:
        raise ValueError(f"CSV is missing one of columns: {', '.join(names)}")
    return None


def read_track_csv(path: str | Path, fps: float) -> list[Observation]:
    """Read a normalized TrackNet/export CSV.

    Required columns: x,y and either time_ms or frame.  Accepted aliases make
    common TrackNet exports easy to adapt: X/Y, Visibility, Frame, frame_num.
    """
    observations: list[Observation] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            x = _number(row, ("x", "X", "x_coordinate"))
            y = _number(row, ("y", "Y", "y_coordinate"))
            time = _number(row, ("time_ms", "timestamp_ms", "timestamp"), required=False)
            if time is None:
                frame = _number(row, ("frame", "Frame", "frame_num", "FrameNum"))
                time = frame * 1000.0 / fps
            confidence = _number(row, ("confidence", "score", "visibility", "Visibility"), required=False)
            confidence = 1.0 if confidence is None else max(0.0, min(1.0, confidence))
            if confidence > 0:
                observations.append(Observation(round(time), x, y, confidence))
    return sorted(observations, key=lambda item: item.time_ms)


def detect_bright_motion(video_path: str | Path, fps_override: float | None = None) -> list[Observation]:
    """Weak fallback when TrackNet coordinates are unavailable.

    It only detects small bright moving blobs, so it is useful for setup tests,
    not for trusted match decisions.  TrackNet CSV is strongly preferred.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    previous: np.ndarray | None = None
    previous_point: tuple[float, float] | None = None
    output: list[Observation] = []
    frame = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            diff = cv2.absdiff(gray, previous)
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(motion, bright)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
            candidates: list[tuple[float, float, float]] = []
            for index in range(1, count):
                area = stats[index, cv2.CC_STAT_AREA]
                if 1 <= area <= 140:
                    cx, cy = centers[index]
                    candidates.append((float(cx), float(cy), float(area)))
            if candidates:
                if previous_point is None:
                    cx, cy, area = max(candidates, key=lambda c: c[2])
                else:
                    cx, cy, area = min(candidates, key=lambda c: (c[0] - previous_point[0]) ** 2 + (c[1] - previous_point[1]) ** 2)
                previous_point = (cx, cy)
                output.append(Observation(round(frame * 1000 / fps), cx, cy, min(0.35, 0.1 + area / 400)))
        previous = gray
        frame += 1
    cap.release()
    return output


def fuse_tracks(
    track_a: list[Observation],
    track_b: list[Observation],
    calibration_a: CameraCalibration,
    calibration_b: CameraCalibration,
    b_offset_ms: int,
    max_pair_delta_ms: int,
) -> list[FusedObservation]:
    """Fuse observations after mapping both views onto the court floor.

    A flying shuttlecock projects to different apparent floor positions.  At a
    true floor contact the two projections converge, which is deliberately used
    later as landing evidence.
    """
    shifted_b = [Observation(o.time_ms + b_offset_ms, o.x, o.y, o.confidence) for o in track_b]
    fused: list[FusedObservation] = []
    j = 0
    for a in track_a:
        while j + 1 < len(shifted_b) and shifted_b[j + 1].time_ms <= a.time_ms:
            j += 1
        candidates = shifted_b[max(0, j - 1): min(len(shifted_b), j + 2)]
        if not candidates:
            continue
        b = min(candidates, key=lambda item: abs(item.time_ms - a.time_ms))
        if abs(a.time_ms - b.time_ms) > max_pair_delta_ms:
            continue
        ax, ay = calibration_a.image_to_court(a.x, a.y)
        bx, by = calibration_b.image_to_court(b.x, b.y)
        disparity = math.hypot(ax - bx, ay - by)
        weight_a, weight_b = a.confidence, b.confidence
        total = max(weight_a + weight_b, 1e-6)
        fused.append(FusedObservation(
            time_ms=round((a.time_ms + b.time_ms) / 2),
            court_x=(ax * weight_a + bx * weight_b) / total,
            court_y=(ay * weight_a + by * weight_b) / total,
            confidence=min(a.confidence, b.confidence),
            plane_disagreement_m=disparity,
            a_image_xy=(a.x, a.y),
            b_image_xy=(b.x, b.y),
        ))
    return fused


def _signed_distance_to_court(x: float, y: float, width: float, length: float) -> float:
    inside = 0 <= x <= width and 0 <= y <= length
    if inside:
        return min(x, width - x, y, length - y)
    dx = max(0.0, -x, x - width)
    dy = max(0.0, -y, y - length)
    return -math.hypot(dx, dy)


def _turn_score(points: list[FusedObservation], i: int) -> float:
    if i < 3 or i + 3 >= len(points):
        return 0.0
    before = np.array([points[i].court_x - points[i - 3].court_x, points[i].court_y - points[i - 3].court_y])
    after = np.array([points[i + 3].court_x - points[i].court_x, points[i + 3].court_y - points[i].court_y])
    denom = float(np.linalg.norm(before) * np.linalg.norm(after))
    if denom < 1e-7:
        return 0.3
    cosine = float(np.dot(before, after) / denom)
    return max(0.0, min(1.0, (1.0 - cosine) / 2.0))


def detect_landings(
    fused: list[FusedObservation],
    court_width_m: float,
    court_length_m: float,
    max_plane_disagreement_m: float,
    review_margin_m: float,
    min_confidence: float,
) -> list[Event]:
    """Detect likely floor contacts from inter-camera floor-projection convergence."""
    candidates: list[Event] = []
    half_window = 4
    for i in range(half_window, len(fused) - half_window):
        point = fused[i]
        neighborhood = fused[i - half_window: i + half_window + 1]
        local_min = min(item.plane_disagreement_m for item in neighborhood)
        if point.plane_disagreement_m > local_min + 1e-6:
            continue
        if point.plane_disagreement_m > max_plane_disagreement_m:
            continue
        # Ignore clear out-of-scene false minima while preserving likely OUTs.
        if not (-1.5 <= point.court_x <= court_width_m + 1.5 and -1.5 <= point.court_y <= court_length_m + 1.5):
            continue
        agreement = math.exp(-point.plane_disagreement_m / max(max_plane_disagreement_m, 1e-6))
        turn = _turn_score(fused, i)
        confidence = point.confidence * agreement * (0.65 + 0.35 * turn)
        if confidence < min_confidence:
            continue
        signed_distance = _signed_distance_to_court(point.court_x, point.court_y, court_width_m, court_length_m)
        if abs(signed_distance) <= review_margin_m:
            result = "UNKNOWN"
            reason = "near_line"
        elif signed_distance > 0:
            result = "IN"
            reason = "inside_court"
        else:
            result = "OUT"
            reason = "outside_court"
        candidates.append(Event("LANDING", point.time_ms, confidence, {
            "in_out": result,
            "reason": reason,
            "court_xy_m": [round(point.court_x, 3), round(point.court_y, 3)],
            "signed_distance_to_boundary_m": round(signed_distance, 3),
            "camera_projection_disagreement_m": round(point.plane_disagreement_m, 3),
        }))
    # A single contact produces neighboring local minima; preserve best evidence.
    selected: list[Event] = []
    for event in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(abs(event.time_ms - kept.time_ms) > 260 for kept in selected):
            selected.append(event)
    return sorted(selected, key=lambda item: item.time_ms)


def detect_trajectory_touches(
    track: list[Observation],
    min_confidence: float,
    min_separation_ms: int,
) -> list[Event]:
    """Find low-confidence contact candidates from a sharp shuttle trajectory turn.

    This intentionally does not identify a player or a racket. It keeps the
    no-pose MVP fully automatic, while the result is honestly labelled as a
    candidate rather than as a confirmed touch.
    """
    candidates: list[Event] = []
    for i in range(2, len(track) - 2):
        before = np.array([track[i].x - track[i - 2].x, track[i].y - track[i - 2].y])
        after = np.array([track[i + 2].x - track[i].x, track[i + 2].y - track[i].y])
        denom = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denom < 36.0:
            continue
        turn = max(-1.0, min(1.0, float(np.dot(before, after) / denom)))
        if turn > 0.55:
            continue
        point = track[i]
        turn_score = (0.55 - turn) / 1.55
        confidence = min(point.confidence, 1.0) * (0.45 + 0.45 * turn_score)
        if confidence >= min_confidence:
            candidates.append(Event("TOUCH", point.time_ms, confidence, {
                "type": "TRAJECTORY_CHANGE_CANDIDATE",
                "warning": "No pose or racket inference is used in this build.",
                "trajectory_turn_cosine": round(turn, 3),
            }))
    selected: list[Event] = []
    for event in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(abs(event.time_ms - kept.time_ms) > min_separation_ms for kept in selected):
            selected.append(event)
    return sorted(selected, key=lambda item: item.time_ms)


def build_timeline(fused: list[FusedObservation], landings: list[Event], touches: list[Event]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    landing_index = 0
    last_landing: Event | None = None
    touch_index = 0
    active_touches: list[Event] = []
    for point in fused:
        while landing_index < len(landings) and landings[landing_index].time_ms <= point.time_ms:
            last_landing = landings[landing_index]
            landing_index += 1
        while touch_index < len(touches) and touches[touch_index].time_ms <= point.time_ms:
            active_touches.append(touches[touch_index])
            touch_index += 1
        active_touches = [event for event in active_touches if point.time_ms - event.time_ms <= 1000]
        latest_landing = last_landing.details["in_out"] if last_landing else "PENDING"
        timeline.append({
            "time_ms": point.time_ms,
            "in_out": latest_landing,
            "touch_last_1s": bool(active_touches),
            "tracking_confidence": round(point.confidence, 3),
        })
    return timeline


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
