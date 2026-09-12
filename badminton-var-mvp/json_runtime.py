"""Persistent JSON-lines bridge between an application and LiveVarSession.

Each line of stdin is one JSON input object; each line of stdout is exactly one
JSON output object. This lets a mobile/desktop app start the process once and
exchange data without an HTTP API.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from realtime_engine import LiveCourt, LiveThresholds, LiveVarSession


sessions: dict[str, LiveVarSession] = {}
tracknet_worker: Any | None = None


def points(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Exactly four court points are required for each camera.")
    converted = []
    for point in value:
        if isinstance(point, dict):
            converted.append([float(point["x"]), float(point["y"])])
        else:
            converted.append([float(point[0]), float(point[1])])
    return converted


def handle(message: dict[str, Any]) -> dict[str, Any]:
    global tracknet_worker
    action = message.get("action")
    session_id = str(message.get("session_id", ""))
    if action == "start":
        if not session_id:
            raise ValueError("session_id is required.")
        court = LiveCourt(**message.get("court", {}))
        thresholds = LiveThresholds(**message.get("thresholds", {}))
        sessions[session_id] = LiveVarSession(session_id, points(message["camera_a_points"]), points(message["camera_b_points"]), court, thresholds)
        return {"type": "session_started", "session_id": session_id, "model_input": "synchronized TrackNet shuttlecock coordinates"}
    if action == "frame_pair":
        if session_id not in sessions:
            raise ValueError("Unknown session_id. Send a start message first.")
        return sessions[session_id].ingest_frame_pair(int(message["time_ms"]), message.get("camera_a"), message.get("camera_b"))
    if action == "video_chunk":
        if session_id not in sessions:
            raise ValueError("Unknown session_id. Send a start message first.")
        # Convenience mode for a prototype: TrackNet runs on each short local
        # clip, then the observations are synchronized and fed into the same
        # stateful engine. A production app can avoid per-clip process startup
        # by using frame_pair with its own persistent TrackNet worker.
        if tracknet_worker is None:
            from tracknet_worker import TrackNetWorker
            tracknet_worker = TrackNetWorker()
        start = int(message["chunk_start_ms"])
        a_track = tracknet_worker.infer_chunk(message["camera_a_video_path"], start)
        b_track = tracknet_worker.infer_chunk(message["camera_b_video_path"], start)
        max_delta = int(message.get("max_pair_delta_ms", 20))
        verdicts: list[dict[str, Any]] = []
        b_index = 0
        for a in a_track:
            while b_index + 1 < len(b_track) and b_track[b_index + 1]["time_ms"] <= a["time_ms"]:
                b_index += 1
            nearby = b_track[max(0, b_index - 1): min(len(b_track), b_index + 2)]
            if not nearby:
                continue
            b = min(nearby, key=lambda point: abs(point["time_ms"] - a["time_ms"]))
            if abs(a["time_ms"] - b["time_ms"]) <= max_delta:
                verdicts.append(sessions[session_id].ingest_frame_pair(round((a["time_ms"] + b["time_ms"]) / 2), a, b))
        latest = verdicts[-1] if verdicts else sessions[session_id].ingest_frame_pair(start, None, None)
        events = [event for verdict in verdicts for event in verdict["new_events"]]
        return {"type": "chunk_verdict", "session_id": session_id, "latest": latest, "new_events": events}
    if action == "end":
        if session_id not in sessions:
            raise ValueError("Unknown session_id.")
        result = sessions.pop(session_id).close()
        return result
    raise ValueError("action must be start, frame_pair, video_chunk, or end.")


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = handle(message)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            response = {"type": "error", "message": str(error)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
