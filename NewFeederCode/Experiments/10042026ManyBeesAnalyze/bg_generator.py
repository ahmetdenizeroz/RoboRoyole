from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class BackgroundBuilder:
    """
    Build a background image from a video by sampling frames
    and taking the per-pixel median.

    This is usually better than averaging when moving objects
    appear in the scene.
    """

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.background: Optional[np.ndarray] = None

    def _open_video(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        return cap

    def _get_frame_count(self, cap: cv2.VideoCapture) -> int:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError("Could not determine frame count.")
        return frame_count

    def build_background(
        self,
        num_frames: int = 50,
        start_ratio: float = 0.0,
        end_ratio: float = 1.0,
        resize_to: Optional[Tuple[int, int]] = None,
        convert_to_gray: bool = False,
    ) -> np.ndarray:
        """
        Sample frames from the video and build a median background.

        Parameters
        ----------
        num_frames : int
            Number of frames to sample.
        start_ratio : float
            Start position in the video, between 0.0 and 1.0.
        end_ratio : float
            End position in the video, between 0.0 and 1.0.
        resize_to : (width, height) or None
            Resize each frame before processing.
        convert_to_gray : bool
            If True, convert frames to grayscale first.

        Returns
        -------
        np.ndarray
            Median background image.
        """
        if num_frames <= 0:
            raise ValueError("num_frames must be greater than 0.")
        if not (0.0 <= start_ratio < end_ratio <= 1.0):
            raise ValueError(
                "start_ratio and end_ratio must satisfy 0.0 <= start_ratio < end_ratio <= 1.0"
            )

        cap = self._open_video()
        frame_count = self._get_frame_count(cap)

        start_idx = int(frame_count * start_ratio)
        end_idx = int(frame_count * end_ratio) - 1

        if end_idx <= start_idx:
            cap.release()
            raise ValueError("Selected video range is too small.")

        sample_indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)

        frames = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

            if convert_to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError("No valid frames could be read from the video.")

        stack = np.stack(frames, axis=0)
        background = np.median(stack, axis=0).astype(np.uint8)

        self.background = background
        return background

    def save_background(self, output_path: str) -> None:
        if self.background is None:
            raise RuntimeError("Background has not been built yet.")

        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(output_path, self.background)
        if not ok:
            raise IOError(f"Could not save background image to: {output_path}")

    def show_background(self, window_name: str = "Median Background") -> None:
        if self.background is None:
            raise RuntimeError("Background has not been built yet.")

        cv2.imshow(window_name, self.background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()