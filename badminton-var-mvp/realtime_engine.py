"""Stateful, pose-free VAR logic for an app that already runs TrackNet.

The application supplies TrackNet's per-frame shuttlecock coordinates as JSON.
This module has no network dependency and immediately returns JSON-compatible
verdict data for each synchronized pair of camera observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from var_core import (
    CameraCalibration,
    Event,
    FusedObservation,
    Observation,
    detect_landings,
    detect_trajectory_touches,
)


@dataclass(frozen=True)
class LiveCourt:
    width_m: float = 6.10
    length_m: float = 13.40
    line_width_m: float = 0.04


@dataclass(frozen=True)
class LiveThresholds:
    max_plane_disagreement_m: float = 0.35
    line_review_margin_m: float = 0.15
    min_landing_confidence: float = 0.45
    min_touch_confidence: float = 0.40
    min_touch_separation_ms: int = 120
    analysis_buffer_ms: int = 3500
    commit_delay_ms: int = 80


def _point(value: dict[str, Any] | None, time_ms: int) -> Observation | None:
    if not value or value.get("confidence", 0.0) <= 0:
        return None
    try:
        return Observation(time_ms, float(value["x"]), float(value["y"]), min(1.0, max(0.0, float(value["confidence"]))))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("A shuttlecock observation needs x, y, and confidence.") from error


class LiveVarSession:
    """Keeps a short trajectory buffer for one live two-camera match session."""

    def __init__(
        self,
        session_id: str,
        camera_a_points: list[list[float]],
        camera_b_points: list[list[float]],
        court: LiveCourt = LiveCourt(),
        thresholds: LiveThresholds = LiveThresholds(),
    ):
        self.session_id = session_id
        self.court = court
        self.thresholds = thresholds
        self.calibration_a = CameraCalibration(camera_a_points, court.width_m, court.length_m)
        self.calibration_b = CameraCalibration(camera_b_points, court.width_m, court.length_m)
        self.track_a: list[Observation] = []
        self.track_b: list[Observation] = []
        self.fused: list[FusedObservation] = []
        self.events: list[Event] = []
        self._emitted: set[tuple[str, int]] = set()
        self._latest_landing: Event | None = None

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - self.thresholds.analysis_buffer_ms
        self.track_a = [item for item in self.track_a if item.time_ms >= cutoff]
        self.track_b = [item for item in self.track_b if item.time_ms >= cutoff]
        self.fused = [item for item in self.fused if item.time_ms >= cutoff]
        self.events = [item for item in self.events if item.time_ms >= cutoff]

    def _fuse_pair(self, a: Observation, b: Observation) -> FusedObservation:
        ax, ay = self.calibration_a.image_to_court(a.x, a.y)
        bx, by = self.calibration_b.image_to_court(b.x, b.y)
        weight = max(a.confidence + b.confidence, 1e-6)
        return FusedObservation(
            time_ms=round((a.time_ms + b.time_ms) / 2),
            court_x=(ax * a.confidence + bx * b.confidence) / weight,
            court_y=(ay * a.confidence + by * b.confidence) / weight,
            confidence=min(a.confidence, b.confidence),
            plane_disagreement_m=math.hypot(ax - bx, ay - by),
            a_image_xy=(a.x, a.y),
            b_image_xy=(b.x, b.y),
        )

    @staticmethod
    def _event_key(event: Event) -> tuple[str, int]:
        # Event estimates can move a few frames as later evidence arrives.
        return event.type, round(event.time_ms / 120)

    def _new_events(self, now_ms: int) -> list[Event]:
        possible_landings = detect_landings(
            self.fused, self.court.width_m, self.court.length_m,
            self.thresholds.max_plane_disagreement_m, self.thresholds.line_review_margin_m,
            self.thresholds.min_landing_confidence,
        )
        possible_touches = detect_trajectory_touches(
            self.track_a, self.thresholds.min_touch_confidence, self.thresholds.min_touch_separation_ms,
        ) + detect_trajectory_touches(
            self.track_b, self.thresholds.min_touch_confidence, self.thresholds.min_touch_separation_ms,
        )
        fresh: list[Event] = []
        for event in sorted(possible_landings + possible_touches, key=lambda item: item.confidence, reverse=True):
            # Wait for a few future frames before committing a local-minimum event.
            if event.time_ms > now_ms - self.thresholds.commit_delay_ms:
                continue
            key = self._event_key(event)
            if key not in self._emitted:
                self._emitted.add(key)
                self.events.append(event)
                fresh.append(event)
                if event.type == "LANDING" and (self._latest_landing is None or event.time_ms >= self._latest_landing.time_ms):
                    self._latest_landing = event
        return sorted(fresh, key=lambda item: item.time_ms)

    def ingest_frame_pair(
        self,
        time_ms: int,
        camera_a: dict[str, Any] | None,
        camera_b: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Ingest one synchronized TrackNet frame pair and return the live verdict."""
        a, b = _point(camera_a, time_ms), _point(camera_b, time_ms)
        if a:
            self.track_a.append(a)
        if b:
            self.track_b.append(b)
        if a and b:
            self.fused.append(self._fuse_pair(a, b))
        self._trim(time_ms)
        new_events = self._new_events(time_ms) if a and b else []
        active_touches = [event for event in self.events if event.type == "TOUCH" and 0 <= time_ms - event.time_ms <= 1000]
        latest_result = self._latest_landing.details["in_out"] if self._latest_landing else "PENDING"
        current_point = self.fused[-1] if self.fused and self.fused[-1].time_ms == time_ms else None
        return {
            "type": "verdict",
            "session_id": self.session_id,
            "time_ms": time_ms,
            "in_out": latest_result,
            "touch_last_1s": bool(active_touches),
            "tracking": {
                "camera_a_detected": a is not None,
                "camera_b_detected": b is not None,
                "confidence": round(current_point.confidence, 3) if current_point else 0.0,
                "camera_projection_disagreement_m": round(current_point.plane_disagreement_m, 3) if current_point else None,
            },
            "new_events": [event.as_json() for event in new_events],
            "warning": "TOUCH is a trajectory-change candidate only; this build has no pose/racket inference.",
        }

    def close(self) -> dict[str, Any]:
        return {
            "type": "session_closed",
            "session_id": self.session_id,
            "events": [event.as_json() for event in sorted(self.events, key=lambda item: item.time_ms)],
        }
