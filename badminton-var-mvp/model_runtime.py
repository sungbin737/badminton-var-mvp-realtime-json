"""Lazy bootstrap and invocation of the open-source TrackNetV3 runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path


TRACKNET_CHECKPOINT_URL = "https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing"


class TrackNetRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Checkpoints:
    tracknet: Path
    inpaintnet: Path


class TrackNetRuntime:
    """Downloads trusted public checkpoints once, then runs TrackNet inference.

    The TrackNet package itself is installed by requirements.txt. Checkpoints are
    not committed to Git and are cached under MODEL_CACHE_DIR after first use.
    """

    def __init__(self, cache_dir: str | Path | None = None, device: str | None = None):
        self.cache_dir = Path(cache_dir or os.getenv("MODEL_CACHE_DIR", ".model-cache")).resolve()
        self.device = device or os.getenv("TRACKNET_DEVICE", "auto")
        self._lock = threading.Lock()

    def _find_checkpoints(self) -> Checkpoints | None:
        tracks = list(self.cache_dir.rglob("TrackNet_best.pt"))
        inpainters = list(self.cache_dir.rglob("InpaintNet_best.pt"))
        if tracks and inpainters:
            return Checkpoints(tracks[0], inpainters[0])
        return None

    @staticmethod
    def _safe_extract(archive: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (destination / member.filename).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise TrackNetRuntimeError("Unsafe path found in checkpoint archive")
            zipped.extractall(destination)

    def ensure_ready(self) -> Checkpoints:
        with self._lock:
            existing = self._find_checkpoints()
            if existing:
                return existing
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            archive = self.cache_dir / "TrackNetV3_ckpts.zip"
            try:
                subprocess.run(
                    [sys.executable, "-m", "gdown", "--fuzzy", TRACKNET_CHECKPOINT_URL, "--output", str(archive)],
                    check=True, capture_output=True, text=True, timeout=600,
                )
                self._safe_extract(archive, self.cache_dir)
            except (OSError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
                raise TrackNetRuntimeError(
                    "TrackNetV3 checkpoints could not be downloaded. Download the official "
                    "checkpoint archive into MODEL_CACHE_DIR and retry."
                ) from error
            finally:
                if archive.exists():
                    archive.unlink()
            checkpoints = self._find_checkpoints()
            if not checkpoints:
                raise TrackNetRuntimeError("Checkpoint archive did not contain TrackNet_best.pt and InpaintNet_best.pt")
            return checkpoints

    def infer(self, video_file: str | Path, output_dir: str | Path) -> Path:
        checkpoints = self.ensure_ready()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "tracknet.inference.tracknet",
            "--video_file", str(video_file),
            "--tracknet_file", str(checkpoints.tracknet),
            "--inpaintnet_file", str(checkpoints.inpaintnet),
            "--save_dir", str(output),
            "--device", self.device,
        ]
        try:
            run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "TrackNet failed").strip()[-2000:]
            raise TrackNetRuntimeError(f"TrackNet inference failed: {detail}") from error
        except OSError as error:
            raise TrackNetRuntimeError("Could not start TrackNet. Install requirements.txt first.") from error
        candidates = sorted(output.rglob("*_ball.csv"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            detail = (run.stdout or "").strip()[-1000:]
            raise TrackNetRuntimeError(f"TrackNet completed but did not create a *_ball.csv file. {detail}")
        return candidates[-1]
