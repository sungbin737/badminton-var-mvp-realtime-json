"""TrackNetV3 adapter for an application's local video chunks.

The app can import TrackNetWorker, send it a recorded clip from either camera,
and pass each returned observation to json_runtime.py or LiveVarSession.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2

from model_runtime import TrackNetRuntime
from var_core import read_track_csv


class TrackNetWorker:
    """Runs the published TrackNetV3 checkpoint and returns JSON-ready points."""

    def __init__(self, runtime: TrackNetRuntime | None = None):
        self.runtime = runtime or TrackNetRuntime()

    def infer_chunk(self, video_path: str | Path, chunk_start_ms: int = 0) -> list[dict[str, Any]]:
        """Return [{time_ms,x,y,confidence}] for a locally accessible video clip."""
        source = Path(video_path)
        capture = cv2.VideoCapture(str(source))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        capture.release()
        if fps <= 0:
            raise ValueError(f"Cannot read FPS from {source}")
        with tempfile.TemporaryDirectory(prefix="tracknet-chunk-") as temporary:
            csv_path = self.runtime.infer(source, temporary)
            observations = read_track_csv(csv_path, fps)
        return [
            {"time_ms": point.time_ms + chunk_start_ms, "x": point.x, "y": point.y, "confidence": point.confidence}
            for point in observations
        ]
