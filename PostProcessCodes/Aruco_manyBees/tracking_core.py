from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from kalman_filter import KalmanFilter
from single_bee_detector_core import (
    AdaptiveRollingBackground,
    SingleBeeDetector,
    arena_bbox_from_mask,
    SingleBeeDetectorSettings,
    default_settings,
    imread_unicode,
    load_settings_txt,
)


Point = Tuple[float, float]
Coord = Tuple[int, int]


class TrackingEngine:
    """
    Single-bee tracking engine.

    Pipeline
    --------
    video frame -> SingleBeeDetector shared core -> one raw bee detection
    -> optional Kalman smoothing/prediction -> raw/filtered coordinate histories.

    This engine intentionally keeps the old coordinate-file convention by using
    marker_ids = [1]. The output columns therefore remain ID_1_X, ID_1_Y,
    ID_1_Ang, so the analysis tool can read old and new experiments with the
    same parser.
    """

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        self.video_path = self.settings["video_path"]

        self.settings_path = (
            self.settings.get("single_bee_settings_path")
            or self.settings.get("settings_preset_path")
            or self.settings.get("detector_settings_path")
        )
        if self.settings_path:
            self.detector_settings = load_settings_txt(self.settings_path)
        else:
            maybe_settings = self.settings.get("single_bee_detector_settings")
            self.detector_settings = maybe_settings if isinstance(maybe_settings, SingleBeeDetectorSettings) else default_settings()

        self.background_path = self.settings.get("background_path") or ""
        self.start_time_s = float(self.settings.get("start_time_s", 0.0))
        self.end_time_s = float(self.settings.get("end_time_s", 0.0))
        self.fps = float(self.settings.get("fps", 30.0) or 30.0)
        self.start_frame = max(0, int(round(self.start_time_s * self.fps)))
        self.end_frame = max(self.start_frame, int(round(self.end_time_s * self.fps)))

        self.arenas = list(self.settings.get("arenas", []))
        self.stimulus_areas = list(self.settings.get("stimulus_areas", []))
        self.pixel_to_mm = float(self.settings.get("pixel_to_mm", 1.0))

        blob_detection_enabled = self.settings.get("blob_detection_enabled", True)
        if not blob_detection_enabled:
            self.detector_settings.aruco.detect_mode = "full_frame"
            self.detector_settings.aruco.enabled = True
            custom_dict = self.settings.get("aruco_dict_path", "")
            if custom_dict:
                self.detector_settings.aruco.dict_path = custom_dict
        else:
            self.detector_settings.aruco.detect_mode = "arena"

        self.marker_ids = [1]
        if self.detector_settings.aruco.enabled and self.detector_settings.aruco.dict_path:
            dict_path = self.detector_settings.aruco.dict_path
            import os
            real_dict_path = dict_path
            if not os.path.isabs(real_dict_path):
                real_dict_path = os.path.join(os.path.dirname(__file__), dict_path)
            if os.path.exists(real_dict_path):
                with open(real_dict_path, "r", encoding="utf-8") as f:
                    parsed_ids = []
                    for line in f:
                        line = line.strip()
                        if line.upper().startswith("ID ") and ":" in line:
                            try:
                                id_str = line.split(":")[0].split(" ")[1]
                                parsed_ids.append(int(id_str))
                            except ValueError:
                                pass
                    if parsed_ids:
                        self.marker_ids = sorted(list(set(parsed_ids)))

        self.id_to_index = {m_id: i for i, m_id in enumerate(self.marker_ids)}

        self.kalman_enabled = bool(self.detector_settings.kalman.enabled)
        self.max_missed_frames = int(self.detector_settings.kalman.max_missed_frames)
        self.kalman_process_noise = float(self.detector_settings.kalman.process_noise)
        self.kalman_measurement_noise = float(self.detector_settings.kalman.measurement_noise)

        # Kept for old info-file/GUI compatibility. Morphology lives inside the
        # detector settings now, not in TrackingEngine.
        self.dilate_kernel_size = int(self.detector_settings.morphology.dilate_kernel)
        self.dilate_iterations = int(self.detector_settings.morphology.dilate_iterations)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if cap_fps > 0 and not self.settings.get("fps"):
            self.fps = cap_fps
            self.start_frame = max(0, int(round(self.start_time_s * self.fps)))
            self.end_frame = max(self.start_frame, int(round(self.end_time_s * self.fps)))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if self.end_frame <= self.start_frame:
            self.end_frame = self.frame_count if self.frame_count > self.start_frame else self.start_frame
        if self.frame_count > 0:
            self.end_frame = min(self.end_frame, self.frame_count)

        self.raw_angles: List[List[float]] = []
        self.filtered_angles: List[List[float]] = []
        self.filtered_is_prediction: List[List[int]] = []
        self.detection_area_history: List[float] = []
        self.candidate_count_history: List[int] = []
        self.last_processed_frame: Optional[np.ndarray] = None

        # Realtime output paths. These are created at the beginning of
        # run_tracking(), so long recordings leave usable coordinate files even
        # if the process/computer stops before plots/videos are generated.
        self.output_dir_path: Optional[str] = None
        self.meta_path: Optional[str] = None
        self.raw_coords_path: Optional[str] = None
        self.filtered_coords_path: Optional[str] = None
        self.progress_path: Optional[str] = None
        self.used_settings_copy_path: Optional[str] = None

        self.stream_output_enabled = bool(self.settings.get("stream_output_realtime", True))
        self.stream_flush_interval = max(1, int(self.settings.get("stream_flush_interval", 100)))
        self.stream_fsync_interval = max(self.stream_flush_interval, int(self.settings.get("stream_fsync_interval", 1000)))

        # Speed option: run real image detection only on selected frames.
        # Value 1 means every frame. Value 5 means frames 1, 5, 10, 15, ...
        # are detection frames; skipped frames are saved as raw missing rows and
        # filtered Kalman predictions when available.
        self.detect_every_n_frames = max(1, int(self.settings.get("detect_every_n_frames", 1)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_tracking(self, progress_callback=None):
        total_frames = max(0, self.end_frame - self.start_frame)
        if total_frames <= 0:
            return [], []

        arena_mask = self._create_shape_mask(self.arenas)
        if cv2.countNonZero(arena_mask) == 0:
            # Safety fallback: if no arena was supplied, allow the full frame.
            arena_mask[:, :] = 255

        detector = SingleBeeDetector(self.detector_settings)
        background_mode = str(self.detector_settings.preprocess.background_mode).strip().lower()

        adaptive_crop_bbox: Optional[Tuple[int, int, int, int]] = None
        if bool(self.detector_settings.preprocess.crop_to_arena_bbox):
            adaptive_crop_bbox = arena_bbox_from_mask(arena_mask, (self.frame_height, self.frame_width), margin=2)

        static_background = None
        if background_mode == "static":
            if not self.background_path:
                raise RuntimeError("Detector preset uses static background, but no background_path was provided.")
            static_background = imread_unicode(self.background_path, cv2.IMREAD_UNCHANGED)
            if static_background is None:
                raise RuntimeError(f"Could not load static background image: {self.background_path}")
            bg_h, bg_w = static_background.shape[:2]
            if (bg_w, bg_h) != (self.frame_width, self.frame_height):
                raise RuntimeError(
                    f"Background size {bg_w}x{bg_h} does not match video frame size "
                    f"{self.frame_width}x{self.frame_height}."
                )

        adaptive_bg: Optional[AdaptiveRollingBackground] = None
        if background_mode == "adaptive":
            adaptive_bg = AdaptiveRollingBackground(
                channel=self.detector_settings.preprocess.channel,
                max_history=self.detector_settings.preprocess.adaptive_history_length,
                method=self.detector_settings.preprocess.adaptive_background_method,
            )
            self._preload_adaptive_background(adaptive_bg, adaptive_crop_bbox)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        raw_coords_history: List[List[Coord]] = []
        filtered_coords_history: List[List[Coord]] = []
        self.raw_angles = []
        self.filtered_angles = []
        self.filtered_is_prediction = []
        self.detection_area_history = []
        self.candidate_count_history = []

        raw_file = None
        filtered_file = None
        if self.stream_output_enabled:
            raw_file, filtered_file = self._prepare_realtime_outputs(total_frames)

        kfs: Dict[int, KalmanFilter] = {}
        missed: Dict[int, int] = {m_id: 0 for m_id in self.marker_ids}
        last_angle_deg: Dict[int, float] = {m_id: -1.0 for m_id in self.marker_ids}
        last_filtered_pos: Dict[int, Optional[Point]] = {m_id: None for m_id in self.marker_ids}

        for local_index in range(total_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            self.last_processed_frame = frame.copy()

            should_detect = (
                self.detect_every_n_frames <= 1
                or local_index == 0
                or ((local_index + 1) % self.detect_every_n_frames == 0)
            )

            result = None
            if should_detect:
                bg_channel = None
                bg_history_count = 0
                if adaptive_bg is not None:
                    bg_channel = adaptive_bg.get_background()
                    bg_history_count = adaptive_bg.history_count

                # Note: SingleBeeDetector handles multiple IDs internally now,
                # so we pass previous_position=None
                result = detector.detect(
                    frame,
                    background_bgr=static_background,
                    background_channel=bg_channel,
                    background_history_count=bg_history_count,
                    arena_mask=arena_mask,
                    previous_position=None,
                )

            if adaptive_bg is not None:
                adaptive_bg.update(self._crop_frame(frame, adaptive_crop_bbox))

            detection_attempted = result is not None
            
            # Map detections to markers
            detections_by_id = {}
            if result is not None:
                self.detection_area_history.append(float(result.area))
                self.candidate_count_history.append(len(result.candidates))
                for cand in result.selected_candidates:
                    if cand.aruco_id is not None:
                        detections_by_id[cand.aruco_id] = cand
                    else:
                        # Fallback for single bee or unassigned
                        detections_by_id[self.marker_ids[0]] = cand
            else:
                self.detection_area_history.append(0.0)
                self.candidate_count_history.append(0)

            frame_raw_pos = []
            frame_raw_ang = []
            frame_filt_pos = []
            frame_filt_ang = []
            frame_pred_flag = []

            for m_id in self.marker_ids:
                cand = detections_by_id.get(m_id)
                detected_now = cand is not None

                if detected_now:
                    raw_pos = (int(round(cand.centroid[0])), int(round(cand.centroid[1])))
                    raw_ang = float(cand.major_axis_angle_deg)
                else:
                    raw_pos = (-1, -1)
                    raw_ang = -1.0

                if not self.kalman_enabled:
                    if detected_now:
                        filt_pos = raw_pos
                        filt_ang = raw_ang
                        pred_flag = 0
                        last_filtered_pos[m_id] = (float(filt_pos[0]), float(filt_pos[1]))
                        last_angle_deg[m_id] = raw_ang
                    else:
                        filt_pos = (-1, -1)
                        filt_ang = -1.0
                        pred_flag = 0
                        last_filtered_pos[m_id] = None
                else:
                    kf = kfs.get(m_id)
                    if detected_now:
                        if kf is None:
                            kf = KalmanFilter(
                                dt=1.0 / max(self.fps, 1e-9),
                                process_noise=self.kalman_process_noise,
                                measurement_noise=self.kalman_measurement_noise,
                            )
                            kfs[m_id] = kf
                            fx, fy = kf.update(cand.centroid[0], cand.centroid[1])
                        else:
                            kf.predict()
                            fx, fy = kf.update(cand.centroid[0], cand.centroid[1])
                        filt_pos = (int(round(fx)), int(round(fy)))
                        filt_ang = raw_ang
                        pred_flag = 0
                        missed[m_id] = 0
                        last_angle_deg[m_id] = raw_ang
                        last_filtered_pos[m_id] = (float(fx), float(fy))
                    else:
                        should_count_as_missed = detection_attempted
                        if kf is not None and (missed[m_id] < self.max_missed_frames or not should_count_as_missed):
                            fx, fy = kf.predict()
                            filt_pos = (int(round(fx)), int(round(fy)))
                            filt_ang = float(last_angle_deg[m_id])
                            pred_flag = 1
                            if should_count_as_missed:
                                missed[m_id] += 1
                            last_filtered_pos[m_id] = (float(fx), float(fy))
                        else:
                            filt_pos = (-1, -1)
                            filt_ang = -1.0
                            pred_flag = 0
                            if should_count_as_missed:
                                missed[m_id] += 1
                            last_filtered_pos[m_id] = None
                            if should_count_as_missed and missed[m_id] > self.max_missed_frames:
                                if m_id in kfs:
                                    del kfs[m_id]

                frame_raw_pos.append(raw_pos)
                frame_raw_ang.append(raw_ang)
                frame_filt_pos.append(filt_pos)
                frame_filt_ang.append(filt_ang)
                frame_pred_flag.append(pred_flag)

            raw_coords_history.append(frame_raw_pos)
            filtered_coords_history.append(frame_filt_pos)
            self.raw_angles.append(frame_raw_ang)
            self.filtered_angles.append(frame_filt_ang)
            self.filtered_is_prediction.append(frame_pred_flag)

            if raw_file is not None and filtered_file is not None:
                self._append_realtime_rows(
                    raw_file=raw_file,
                    filtered_file=filtered_file,
                    raw_pos=frame_raw_pos,
                    raw_ang=frame_raw_ang,
                    filt_pos=frame_filt_pos,
                    filt_ang=frame_filt_ang,
                    local_index=local_index,
                    total_frames=total_frames,
                )

            if progress_callback is not None:
                progress_callback(int(100 * (local_index + 1) / total_frames))

        if raw_file is not None and filtered_file is not None:
            self._close_realtime_outputs(raw_file, filtered_file, status="finished", processed_frames=len(raw_coords_history), total_frames=total_frames)

        cap.release()
        return raw_coords_history, filtered_coords_history

    def analyze_results(self, filtered_coords):
        # Kept as a compatibility hook for the GUI. Detailed analysis is done by
        # analysis_core.py after files are written.
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_output_dir(self) -> Path:
        base_name = str(self.settings.get("exp_name") or "single_bee_experiment").strip() or "single_bee_experiment"
        parent = self.settings.get("output_dir") or os.path.dirname(self.video_path) or "."
        output_dir = Path(parent) / f"{base_name}_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_path = str(output_dir)
        return output_dir

    def _prepare_realtime_outputs(self, total_frames: int):
        output_dir = self._resolve_output_dir()
        base_name = str(self.settings.get("exp_name") or "single_bee_experiment").strip() or "single_bee_experiment"

        self.meta_path = str(output_dir / f"{base_name}_info.txt")
        self.raw_coords_path = str(output_dir / f"{base_name}_coordinates_raw.txt")
        self.filtered_coords_path = str(output_dir / f"{base_name}_coordinates_filtered.txt")
        self.progress_path = str(output_dir / f"{base_name}_tracking_progress.txt")

        self._write_realtime_info_file(total_frames=total_frames, status="running")
        self._copy_used_settings(output_dir, base_name)

        header = "\t".join([f"ID_{marker_id}_X\tID_{marker_id}_Y\tID_{marker_id}_Ang" for marker_id in self.marker_ids])
        raw_file = open(self.raw_coords_path, "w", encoding="utf-8", buffering=1)
        filtered_file = open(self.filtered_coords_path, "w", encoding="utf-8", buffering=1)
        raw_file.write(header + "\n")
        filtered_file.write(header + "\n")
        raw_file.flush()
        filtered_file.flush()
        self._safe_fsync(raw_file)
        self._safe_fsync(filtered_file)
        self._write_progress_file(status="running", processed_frames=0, total_frames=total_frames)
        return raw_file, filtered_file

    def _write_realtime_info_file(self, *, total_frames: int, status: str) -> None:
        if not self.meta_path:
            return
        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write("# Video Info\n")
            f.write(f"video_path\t{self.video_path}\n")
            f.write(f"fps\t{self.fps:.4f}\n")
            f.write(f"frame_width\t{self.frame_width}\n")
            f.write(f"frame_height\t{self.frame_height}\n")
            f.write(f"start_time_str\t{self.settings.get('start_time_str', self.start_time_s)}\n")
            f.write(f"end_time_str\t{self.settings.get('end_time_str', self.end_time_s)}\n")
            f.write(f"start_time_s\t{self.start_time_s:.6f}\n")
            f.write(f"end_time_s\t{self.end_time_s:.6f}\n")
            f.write(f"start_frame\t{self.start_frame}\n")
            f.write(f"end_frame\t{self.end_frame}\n")
            f.write(f"planned_total_frames\t{total_frames}\n")
            f.write(f"tracking_status\t{status}\n\n")

            f.write("# Inputs\n")
            f.write(f"single_bee_settings_path\t{self.settings_path or ''}\n")
            f.write(f"background_path\t{self.background_path}\n")
            f.write("tracking_method\tSingle Bee Detector\n")
            f.write("stream_output_realtime\ttrue\n")
            f.write(f"detect_every_n_frames\t{self.detect_every_n_frames}\n")
            f.write("detect_every_n_frames_rule\t1 means every frame; N means frames 1,N,2N,3N,... are real detection frames\n\n")

            f.write("# Marker IDs\n")
            f.write("marker_ids\t" + ",".join(str(x) for x in self.marker_ids) + "\n")
            f.write("single_bee_id\tID_1\n")
            f.write("raw_missing_rule\t-1,-1,-1 means no real detection on that frame\n")
            f.write("filtered_prediction_rule\tIf raw is missing but filtered exists, that point was generated by Kalman prediction\n\n")

            f.write("# Scale\n")
            f.write(f"pixel_to_mm\t{self.pixel_to_mm:.8f}\n\n")

            ds = self.detector_settings
            f.write("# Single Bee Detector Settings Summary\n")
            f.write(f"background_mode\t{ds.preprocess.background_mode}\n")
            f.write(f"adaptive_history_length\t{ds.preprocess.adaptive_history_length}\n")
            f.write(f"adaptive_background_method\t{ds.preprocess.adaptive_background_method}\n")
            f.write(f"channel\t{ds.preprocess.channel}\n")
            f.write(f"difference_mode\t{ds.preprocess.difference_mode}\n")
            f.write(f"blur_kernel\t{ds.preprocess.blur_kernel}\n")
            f.write(f"detection_scale\t{ds.preprocess.detection_scale}\n")
            f.write(f"refine_on_original\t{ds.preprocess.refine_on_original}\n")
            f.write(f"crop_to_arena_bbox\t{ds.preprocess.crop_to_arena_bbox}\n")
            f.write(f"refine_padding_px\t{ds.preprocess.refine_padding_px}\n")
            f.write(f"threshold_mode\t{ds.threshold.mode}\n")
            f.write(f"manual_threshold\t{ds.threshold.manual_threshold}\n")
            f.write(f"min_area\t{ds.blob_filter.min_area}\n")
            f.write(f"max_area\t{ds.blob_filter.max_area}\n")
            f.write(f"candidate_selection_method\t{ds.candidate_selection.method}\n\n")

            f.write("# Kalman Settings\n")
            f.write(f"kalman_enabled\t{self.kalman_enabled}\n")
            f.write(f"kalman_max_missed_frames\t{self.max_missed_frames}\n")
            f.write(f"kalman_process_noise\t{self.kalman_process_noise}\n")
            f.write(f"kalman_measurement_noise\t{self.kalman_measurement_noise}\n\n")

            f.write("# Arenas\n")
            f.write("id\tshape\tcategory\tgeom\n")
            for shape in self.arenas:
                f.write(f"{shape.get('id', '')}\t{shape.get('shape', '')}\t{shape.get('category', '')}\t{shape.get('geom', '')}\n")
            f.write("\n# Stimulus Areas\n")
            f.write("id\tshape\tcategory\tgeom\n")
            for shape in self.stimulus_areas:
                f.write(f"{shape.get('id', '')}\t{shape.get('shape', '')}\t{shape.get('category', '')}\t{shape.get('geom', '')}\n")

    def _copy_used_settings(self, output_dir: Path, base_name: str) -> None:
        if self.settings_path and os.path.exists(self.settings_path):
            self.used_settings_copy_path = str(output_dir / f"{base_name}_used_single_bee_settings.txt")
            try:
                shutil.copyfile(self.settings_path, self.used_settings_copy_path)
            except Exception:
                pass

    def _append_realtime_rows(
        self,
        *,
        raw_file,
        filtered_file,
        raw_pos: List[Coord],
        raw_ang: List[float],
        filt_pos: List[Coord],
        filt_ang: List[float],
        local_index: int,
        total_frames: int,
    ) -> None:
        raw_line = "\t".join(f"{p[0]}\t{p[1]}\t{float(a):.6f}" for p, a in zip(raw_pos, raw_ang))
        filt_line = "\t".join(f"{p[0]}\t{p[1]}\t{float(a):.6f}" for p, a in zip(filt_pos, filt_ang))
        raw_file.write(raw_line + "\n")
        filtered_file.write(filt_line + "\n")

        processed = local_index + 1
        if processed % self.stream_flush_interval == 0:
            raw_file.flush()
            filtered_file.flush()
            self._write_progress_file(status="running", processed_frames=processed, total_frames=total_frames)

        if processed % self.stream_fsync_interval == 0:
            self._safe_fsync(raw_file)
            self._safe_fsync(filtered_file)

    def _close_realtime_outputs(self, raw_file, filtered_file, *, status: str, processed_frames: int, total_frames: int) -> None:
        try:
            raw_file.flush()
            filtered_file.flush()
            self._safe_fsync(raw_file)
            self._safe_fsync(filtered_file)
        finally:
            try:
                raw_file.close()
            except Exception:
                pass
            try:
                filtered_file.close()
            except Exception:
                pass
        self._write_progress_file(status=status, processed_frames=processed_frames, total_frames=total_frames)
        self._write_realtime_info_file(total_frames=total_frames, status=status)

    @staticmethod
    def _safe_fsync(file_obj) -> None:
        try:
            os.fsync(file_obj.fileno())
        except Exception:
            pass

    def _write_progress_file(self, *, status: str, processed_frames: int, total_frames: int) -> None:
        if not self.progress_path:
            return
        percent = 0.0 if total_frames <= 0 else 100.0 * float(processed_frames) / float(total_frames)
        last_abs_frame = self.start_frame + max(0, processed_frames - 1)
        tmp_path = self.progress_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(f"status\t{status}\n")
            f.write(f"processed_frames\t{processed_frames}\n")
            f.write(f"total_frames\t{total_frames}\n")
            f.write(f"percent\t{percent:.4f}\n")
            f.write(f"start_frame\t{self.start_frame}\n")
            f.write(f"last_absolute_frame\t{last_abs_frame}\n")
            f.write(f"raw_coordinates_path\t{self.raw_coords_path or ''}\n")
            f.write(f"filtered_coordinates_path\t{self.filtered_coords_path or ''}\n")
        try:
            os.replace(tmp_path, self.progress_path)
        except Exception:
            pass

    def _preload_adaptive_background(
        self,
        adaptive_bg: AdaptiveRollingBackground,
        crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Load previous frames into the adaptive background at start_frame."""
        history_len = max(1, int(self.detector_settings.preprocess.adaptive_history_length))
        preload_start = max(0, self.start_frame - history_len)
        if self.start_frame <= preload_start:
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, preload_start)
        for _idx in range(preload_start, self.start_frame):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            adaptive_bg.update(self._crop_frame(frame, crop_bbox))
        cap.release()

    @staticmethod
    def _crop_frame(frame: np.ndarray, crop_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if crop_bbox is None:
            return frame
        x, y, w, h = [int(v) for v in crop_bbox]
        return frame[y:y + h, x:x + w]

    def _create_shape_mask(self, shapes: List[dict]) -> np.ndarray:
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        for shape_data in shapes:
            if not shape_data or "geom" not in shape_data:
                continue
            geom = shape_data["geom"]
            shape = shape_data.get("shape")
            if shape == "circle":
                cv2.circle(mask, (int(geom[0]), int(geom[1])), int(geom[2]), 255, -1)
            elif shape == "rect":
                x, y, w, h = [int(v) for v in geom]
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            elif shape == "poly":
                pts = np.array(geom, dtype=np.int32)
                if pts.size > 0:
                    cv2.fillPoly(mask, [pts], 255)
        return mask

    def _is_point_in_shape(self, point, shape_data):
        x, y = point
        shape = shape_data.get("shape")
        geom = shape_data.get("geom")
        if not geom:
            return False
        if shape == "circle":
            return (x - geom[0]) ** 2 + (y - geom[1]) ** 2 < geom[2] ** 2
        if shape == "rect":
            return geom[0] <= x <= geom[0] + geom[2] and geom[1] <= y <= geom[1] + geom[3]
        if shape == "poly":
            return cv2.pointPolygonTest(np.array(geom, dtype=np.int32), (float(x), float(y)), False) >= 0
        return False
