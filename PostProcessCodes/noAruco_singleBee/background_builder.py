import cv2
import numpy as np


class BackgroundBuilder:
    """
    Builds or loads a static background image for a video.

    In the new single-bee workflow this class is optional: it is only needed
    when the detector settings preset uses background_mode = static. For
    background_mode = adaptive, the tracking core uses rolling previous-frame
    background from single_bee_detector_core.py instead.

    Main features
    -------------
    - Build background from a video using median of sampled frames
    - Load an already saved background image
    - Save background image
    - Debug one minute against the background
    - Debug a range of times with a chosen step
    """

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.background = None
        self.background_gray = None
        self.video_info = self._read_video_info()

    # ------------------------------------------------------------------
    # Video info
    # ------------------------------------------------------------------
    def _read_video_info(self) -> dict:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": 0.0,
        }
        cap.release()

        if info["fps"] > 0:
            info["duration_s"] = info["frame_count"] / info["fps"]

        return info

    def get_video_info(self) -> dict:
        return dict(self.video_info)

    # ------------------------------------------------------------------
    # Background creation / loading
    # ------------------------------------------------------------------
    def _get_sample_indices(
        self,
        num_frames: int,
        start_frame: int = 0,
        end_frame: int = None,
    ) -> np.ndarray:
        total_frames = self.video_info["frame_count"]

        if end_frame is None:
            end_frame = total_frames

        start_frame = max(0, int(start_frame))
        end_frame = min(total_frames, int(end_frame))

        if end_frame <= start_frame:
            raise ValueError("end_frame must be greater than start_frame.")

        available = end_frame - start_frame
        num_frames = max(1, min(int(num_frames), available))

        indices = np.linspace(start_frame, end_frame - 1, num=num_frames, dtype=int)
        indices = np.unique(indices)

        if len(indices) == 0:
            raise RuntimeError("No valid frame indices could be generated.")

        return indices

    def _set_background(self, bg: np.ndarray) -> None:
        if bg is None:
            raise ValueError("Background image is None.")

        if bg.ndim == 2:
            self.background = bg.copy()
            self.background_gray = bg.copy()
        elif bg.ndim == 3:
            self.background = bg.copy()
            self.background_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Background image must be grayscale or BGR.")

    def build(
        self,
        num_frames: int,
        start_frame: int = 0,
        end_frame: int = None,
        grayscale: bool = False,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Build background image using median pixel value across sampled frames.
        """
        sample_indices = self._get_sample_indices(
            num_frames=num_frames,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        frames = []

        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

            if progress_callback is not None:
                percent = int(100 * (i + 1) / len(sample_indices))
                progress_callback(percent)

        cap.release()

        if not frames:
            raise RuntimeError("No frames could be read for background creation.")

        stacked = np.stack(frames, axis=0)
        background = np.median(stacked, axis=0).astype(np.uint8)

        self._set_background(background)
        return self.background

    def load_background(self, background_path: str, grayscale: bool = False) -> np.ndarray:
        """
        Load a previously saved background image.

        Parameters
        ----------
        background_path : str
            Path to the background image file.
        grayscale : bool
            If True, read the file directly as grayscale.
            If False, read as color (BGR).
        """
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        bg = cv2.imread(str(background_path), flag)

        if bg is None:
            raise RuntimeError(f"Could not load background image: {background_path}")

        self._set_background(bg)
        return self.background

    def save(self, output_path: str) -> None:
        if self.background is None:
            raise RuntimeError("No background image exists yet. Build or load one first.")

        ok = cv2.imwrite(str(output_path), self.background)
        if not ok:
            raise RuntimeError(f"Could not save background image to: {output_path}")

    def build_and_save(
        self,
        output_path: str,
        num_frames: int,
        start_frame: int = 0,
        end_frame: int = None,
        grayscale: bool = False,
        progress_callback=None,
    ) -> np.ndarray:
        bg = self.build(
            num_frames=num_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            grayscale=grayscale,
            progress_callback=progress_callback,
        )
        self.save(output_path)
        return bg

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------
    def _read_frame_at_second(self, second: float) -> np.ndarray:
        fps = self.video_info["fps"]
        if fps <= 0:
            raise RuntimeError("Video FPS is invalid.")

        frame_idx = int(second * fps)
        frame_idx = max(0, min(frame_idx, self.video_info["frame_count"] - 1))

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame at second {second:.2f}.")

        return frame

    def _compute_debug_images(self, second: float) -> dict:
        if self.background is None or self.background_gray is None:
            raise RuntimeError("Background is not available. Build or load it first.")

        if second < 0 or second > self.video_info["duration_s"]:
            raise ValueError(
                f"Requested second {second:.2f} is outside video duration "
                f"(0 to {self.video_info['duration_s']:.2f} s)."
            )

        frame_bgr = self._read_frame_at_second(second)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame_gray, self.background_gray)
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return {
            "frame_bgr": frame_bgr,
            "background": self.background,
            "diff": diff,
            "binary": binary,
            "second": second,
            "mean_diff": float(diff.mean()),
            "max_diff": int(diff.max()),
            "nonzero_binary": int(cv2.countNonZero(binary)),
        }

    # ------------------------------------------------------------------
    # Debug methods
    # ------------------------------------------------------------------
    def _show_resized(self, window_name: str, image, width: int = 600, height: int = 400) -> None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.imshow(window_name, image)
        
    def debug_minute(
        self,
        minute: float,
        show_binary: bool = True,
        window_prefix: str = "Background Debug",
    ) -> None:
        """
        Show debug windows for the frame at the given minute.
        """
        second = float(minute) * 60.0
        data = self._compute_debug_images(second=second)

        self._show_resized(f"{window_prefix} - Frame", data["frame_bgr"], 600, 400)
        self._show_resized(f"{window_prefix} - Background", data["background"], 600, 400)
        self._show_resized(f"{window_prefix} - Absolute Diff", data["diff"], 600, 400)

        if show_binary:
            self._show_resized(f"{window_prefix} - Binary Diff", data["binary"], 600, 400)

        print(
            f"[DEBUG] minute={minute:.2f}, second={data['second']:.2f}, "
            f"mean_diff={data['mean_diff']:.2f}, max_diff={data['max_diff']}, "
            f"nonzero_binary={data['nonzero_binary']}"
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def debug_range(
        self,
        start_minute: float,
        end_minute: float,
        step_seconds: float,
        show_binary: bool = True,
        stop_on_key: bool = True,
        window_prefix: str = "Background Debug Range",
    ) -> list:
        """
        Debug a time range by checking one frame every step_seconds.

        Parameters
        ----------
        start_minute : float
            Start minute of the scan.
        end_minute : float
            End minute of the scan.
        step_seconds : float
            Time step in seconds between checks.
        show_binary : bool
            If True, also show binary diff window.
        stop_on_key : bool
            If True, stops when user presses q or ESC.
        window_prefix : str
            Prefix for OpenCV window titles.

        Returns
        -------
        list of dict
            Debug summary for each sampled time point.
        """
        if step_seconds <= 0:
            raise ValueError("step_seconds must be greater than 0.")

        start_second = float(start_minute) * 60.0
        end_second = float(end_minute) * 60.0

        if end_second < start_second:
            raise ValueError("end_minute must be greater than or equal to start_minute.")

        duration_s = self.video_info["duration_s"]
        start_second = max(0.0, start_second)
        end_second = min(duration_s, end_second)

        results = []
        current_second = start_second

        while current_second <= end_second + 1e-9:
            data = self._compute_debug_images(second=current_second)
            results.append({
                "second": data["second"],
                "minute": data["second"] / 60.0,
                "mean_diff": data["mean_diff"],
                "max_diff": data["max_diff"],
                "nonzero_binary": data["nonzero_binary"],
            })

            frame_display = data["frame_bgr"].copy()
            text = (
                f"t={data['second']:.1f}s  "
                f"mean={data['mean_diff']:.2f}  "
                f"max={data['max_diff']}  "
                f"nz={data['nonzero_binary']}"
            )
            cv2.putText(
                frame_display,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            self._show_resized(f"{window_prefix} - Frame", data["frame_bgr"], 600, 400)
            self._show_resized(f"{window_prefix} - Background", data["background"], 600, 400)
            self._show_resized(f"{window_prefix} - Absolute Diff", data["diff"], 600, 400)

            if show_binary:
                self._show_resized(f"{window_prefix} - Binary Diff", data["binary"], 600, 400)

            print(
                f"[DEBUG RANGE] second={data['second']:.2f}, minute={data['second']/60.0:.2f}, "
                f"mean_diff={data['mean_diff']:.2f}, max_diff={data['max_diff']}, "
                f"nonzero_binary={data['nonzero_binary']}"
            )

            key = cv2.waitKey(0 if stop_on_key else 1) & 0xFF
            if stop_on_key and key in (27, ord('q'), ord('Q')):
                break

            current_second += step_seconds

        cv2.destroyAllWindows()
        return results