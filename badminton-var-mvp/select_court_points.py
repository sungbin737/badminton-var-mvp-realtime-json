"""Click four court corners in a video frame and save them into config.json.

Order: near-left, near-right, far-right, far-left.  Press U to undo the last
point, R to start over, Enter to save four points, and Esc to cancel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


POINT_LABELS = ["near-left", "near-right", "far-right", "far-left"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select four badminton-court corners")
    parser.add_argument("--config", required=True, help="Configuration JSON to update")
    parser.add_argument("--camera", choices=("a", "b"), required=True, help="Camera whose corners will be saved")
    parser.add_argument("--video", help="Optional video path; otherwise uses camera_a/camera_b.video in config")
    parser.add_argument("--frame", type=int, default=0, help="Frame to show; choose a clear empty-court frame")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum on-screen preview width")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    camera_key = f"camera_{args.camera}"
    video_path = args.video or config[camera_key]["video"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, original = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Cannot read frame {args.frame} from: {video_path}")

    height, width = original.shape[:2]
    scale = min(1.0, args.max_width / width)
    shown_size = (round(width * scale), round(height * scale))
    points: list[tuple[float, float]] = []
    window = "Click court corners: near-left, near-right, far-right, far-left"

    def render() -> np.ndarray:
        shown = cv2.resize(original, shown_size, interpolation=cv2.INTER_AREA) if scale != 1.0 else original.copy()
        for index, (x, y) in enumerate(points):
            px, py = round(x * scale), round(y * scale)
            cv2.circle(shown, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(shown, f"{index + 1}: {POINT_LABELS[index]}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        prompt = "Click: " + (POINT_LABELS[len(points)] if len(points) < 4 else "press Enter to save")
        cv2.rectangle(shown, (0, 0), (min(shown.shape[1], 700), 38), (0, 0, 0), -1)
        cv2.putText(shown, prompt + " | U: undo, R: reset, Esc: cancel", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return shown

    def click(event: int, x: int, y: int, _flags: int, _value: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, click)
    while True:
        cv2.imshow(window, render())
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("Cancelled; configuration was not changed.")
            return
        if key in (ord("u"), ord("U")) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (10, 13) and len(points) == 4:
            config[camera_key]["court_image_points"] = [[round(x, 2), round(y, 2)] for x, y in points]
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            cv2.destroyAllWindows()
            print(f"Saved {camera_key}.court_image_points to {config_path}")
            return


if __name__ == "__main__":
    main()
