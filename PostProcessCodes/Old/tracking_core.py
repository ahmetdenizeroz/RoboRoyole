# title: "Animal Tracking Core"
# date: "10/17/2025"
# author: "Babur Erdem"
# modified: "10/26/2025"
# modified: "11/02/2025" # Added Kalman filter gap-filling
# modified: "11/02/2025" # Added correct filter initialization

import cv2
import numpy as np
import math
from collections import defaultdict


# --- Kalman Filter ---
class KalmanFilter:
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * process_noise_std ** 2
        self.R = np.eye(2, dtype=np.float32) * measurement_noise_std ** 2
        self.P = np.eye(4, dtype=np.float32) * 1000.0
        self.x_hat = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def predict(self):
        self.x_hat = np.dot(self.A, self.x_hat)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x_hat[0:2].flatten().astype(int)

    def update(self, measurement):
        if not self.initialized:
            self.x_hat[0] = measurement[0]
            self.x_hat[1] = measurement[1]
            self.P = np.eye(4, dtype=np.float32) * 10.0
            self.initialized = True
            return self.x_hat[0:2].flatten().astype(int)

        y = measurement.reshape(2, 1) - np.dot(self.H, self.x_hat)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x_hat = self.x_hat + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return self.x_hat[0:2].flatten().astype(int)


# --- Tracking Engine ---
class TrackingEngine:
    def __init__(self, settings):
        self.settings = settings
        self.video_path = settings['video_path']
        self.start_time_s = settings['start_time_s']
        self.end_time_s = settings['end_time_s']
        self.fps = settings.get('fps', 30)
        self.start_frame = int(self.start_time_s * self.fps)
        self.end_frame = int(self.end_time_s * self.fps)
        self.arenas = settings['arenas']
        self.stimulus_areas = settings['stimulus_areas']
        self.pixel_to_mm = settings.get('pixel_to_mm', 1.0)
        self.min_area = settings.get('min_area', 100)
        self.max_area = settings.get('max_area', 10000)

        # --- Tracking parameters ---
        self.tracking_method = settings.get('tracking_method', 'Background Subtraction')
        self.varThreshold = 16  # Fixed default for BG Subtraction

        # Parameters for Blob Detection
        self.blur_kernel_size = settings.get('blur_kernel_size', 5)
        self.blob_min_threshold = settings.get('blob_min_threshold', 50)
        self.blob_max_threshold = settings.get('blob_max_threshold', 220)

        # Defaults for internal processing
        self.morph_kernel_size = 3  # Fixed default

        cap = cv2.VideoCapture(self.video_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def _create_shape_mask(self, shapes):
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        if not shapes: return mask
        for shape_data in shapes:
            geom = shape_data['geom']
            if shape_data['shape'] == 'circle':
                cv2.circle(mask, (geom[0], geom[1]), geom[2], 255, -1)
            elif shape_data['shape'] == 'rect':
                cv2.rectangle(mask, (geom[0], geom[1]), (geom[0] + geom[2], geom[1] + geom[3]), 255, -1)
            elif shape_data['shape'] == 'poly':
                cv2.fillPoly(mask, [np.array(geom, dtype=np.int32)], 255)
        return mask

    def run_tracking(self, progress_callback):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        total_frames = self.end_frame - self.start_frame

        bg_subtractor = None
        detector = None

        # --- Initialize selected tracking method ---
        if self.tracking_method == 'Background Subtraction':
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.varThreshold,
                                                            detectShadows=False)

        elif self.tracking_method == 'Blob Detection':
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = self.blob_min_threshold
            params.maxThreshold = self.blob_max_threshold
            params.thresholdStep = 10
            params.filterByArea = True
            params.minArea = self.min_area
            params.maxArea = self.max_area
            params.filterByColor = False
            params.filterByCircularity = False
            params.filterByInertia = False
            params.filterByConvexity = False
            detector = cv2.SimpleBlobDetector_create(params)

        arena_mask = self._create_shape_mask(self.arenas)
        morph_kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)

        raw_coords_history = []
        filtered_coords_history = []
        kalman_filters = [KalmanFilter(1.0 / self.fps, 1, 10) for _ in self.arenas]
        # Initialize a tracker for the last known good position for each arena
        last_good_pos = [self._get_shape_center(arena) for arena in self.arenas]
        # --- NEW --- Track which filters have been properly initialized
        filters_initialized = [False] * len(self.arenas)

        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_image = None
            detected_centers = []

            # --- Apply selected tracking method ---
            if self.tracking_method == 'Background Subtraction':
                fg_mask = bg_subtractor.apply(frame)
                fg_mask_masked = cv2.bitwise_and(fg_mask, fg_mask, mask=arena_mask)
                thresh = cv2.threshold(fg_mask_masked, 25, 255, cv2.THRESH_BINARY)[1]
                processed_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=1)

                # --- Find Contours for BGSub ---
                contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    area = cv2.contourArea(c)
                    if self.min_area < area < self.max_area:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            detected_centers.append((cx, cy))

            elif self.tracking_method == 'Blob Detection':
                # Apply blur
                blur = cv2.medianBlur(gray, self.blur_kernel_size)
                # Apply arena mask *before* detection
                processed_image = cv2.bitwise_and(blur, blur, mask=arena_mask)

                # --- Use Blob Detector ---
                # Detector works on the blurred grayscale image
                keypoints = detector.detect(processed_image)
                for kp in keypoints:
                    detected_centers.append((int(kp.pt[0]), int(kp.pt[1])))

            # --- Associate centers with arenas (same logic) ---
            current_raw_coords = []
            current_filtered_coords = []
            assigned_centers = [False] * len(detected_centers)
            for i, arena in enumerate(self.arenas):
                best_match, min_dist, best_match_idx = None, float('inf'), -1
                for j, center in enumerate(detected_centers):
                    if assigned_centers[j]: continue
                    if self._is_point_in_shape(center, arena):
                        last_pos = filtered_coords_history[-1][i] if filtered_coords_history and \
                                                                    filtered_coords_history[-1][i][
                                                                        0] != -1 else self._get_shape_center(arena)
                        dist = math.sqrt((center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2)
                        if dist < min_dist:
                            min_dist, best_match, best_match_idx = dist, center, j
                if best_match_idx != -1: assigned_centers[best_match_idx] = True

                raw_pos = best_match if best_match else (-1, -1)
                current_raw_coords.append(raw_pos)

                kf = kalman_filters[i]

                # --- NEW Initialization and Gap-Filling Logic ---
                if raw_pos != (-1, -1):
                    # --- We have a good detection ---

                    # 1. This is either the first detection or a subsequent one.
                    #    In either case, update the filter with this real position.
                    kf.update(np.array(raw_pos, dtype=np.float32))

                    # 2. Set/update the "last good position"
                    last_good_pos[i] = raw_pos

                    # 3. Mark this filter as initialized
                    filters_initialized[i] = True

                    # 4. Get the prediction (which will be based on the update)
                    predicted_pos = kf.predict()
                    current_filtered_coords.append(
                        tuple(predicted_pos) if self._is_point_in_shape(predicted_pos, arena) else (-1, -1))

                else:
                    # --- We have a lost detection (raw_pos is -1, -1) ---

                    if filters_initialized[i]:
                        # 1. We have seen this animal before, so fill the gap
                        #    Use the "last good position" as the measurement
                        kf.update(np.array(last_good_pos[i], dtype=np.float32))

                        # 2. Get the new prediction based on the filled gap
                        predicted_pos = kf.predict()
                        current_filtered_coords.append(
                            tuple(predicted_pos) if self._is_point_in_shape(predicted_pos, arena) else (-1, -1))
                    else:
                        # 1. We have *never* seen this animal.
                        # 2. Do not initialize the filter. Do not predict.
                        # 3. The filtered coordinate remains (-1, -1).
                        current_filtered_coords.append((-1, -1))
                # --- End of New Logic ---

            raw_coords_history.append(current_raw_coords)
            filtered_coords_history.append(current_filtered_coords)

            if progress_callback: progress_callback(int(100 * (frame_count + 1) / total_frames))

        cap.release()
        return raw_coords_history, filtered_coords_history

    # --- analyze_results is now only used internally if needed, not for final output files ---
    def analyze_results(self, filtered_coords):
        num_arenas, num_stim = len(self.arenas), len(self.stimulus_areas)
        total_dist = [0.0] * num_arenas
        avg_dist_from_center = [0.0] * num_arenas
        time_in_stim = [defaultdict(float) for _ in range(num_arenas)]
        entries_to_stim = [defaultdict(int) for _ in range(num_arenas)]
        was_in_stim = [[False] * num_stim for _ in range(num_arenas)]
        entry_events = [[[] for _ in range(num_stim)] for _ in range(num_arenas)]  # Kept for potential future use

        for frame_idx, frame_coords in enumerate(filtered_coords):
            prev_frame_coords = filtered_coords[frame_idx - 1] if frame_idx > 0 else None
            for i in range(num_arenas):
                pos = frame_coords[i]
                if pos[0] == -1:
                    continue
                if prev_frame_coords:
                    prev_pos = prev_frame_coords[i]
                    if prev_pos[0] != -1:
                        dist_px = math.sqrt((pos[0] - prev_pos[0]) ** 2 + (pos[1] - prev_pos[1]) ** 2)
                        total_dist[i] += dist_px * self.pixel_to_mm
                arena_center = self._get_shape_center(self.arenas[i])
                dist_from_center_px = math.sqrt((pos[0] - arena_center[0]) ** 2 + (pos[1] - arena_center[1]) ** 2)
                avg_dist_from_center[i] += dist_from_center_px * self.pixel_to_mm

                # --- Stimulus calculations (kept internally for now) ---
                for j in range(num_stim):
                    is_in = self._is_point_in_shape(pos, self.stimulus_areas[j])
                    if is_in:
                        time_in_stim[i][j] += 1.0 / self.fps
                        if not was_in_stim[i][j]: entries_to_stim[i][j] += 1
                    was_in_stim[i][j] = is_in

        num_frames_tracked = [sum(1 for f in filtered_coords if f[i][0] != -1) for i in range(num_arenas)]
        avg_speed = [(total_dist[i] / ((nf - 1) / self.fps)) if nf > 1 else 0 for i, nf in
                    enumerate(num_frames_tracked)]
        avg_dist_from_center = [avg_dist_from_center[i] / nf if nf > 0 else 0 for i, nf in
                                enumerate(num_frames_tracked)]

        # Return the calculated values, even if not saved directly to files anymore
        return {'avg_speed': avg_speed, 'avg_dist_from_center': avg_dist_from_center,
                'time_in_stim': time_in_stim, 'entries_to_stim': entries_to_stim, 'entry_events': entry_events}

    def _is_point_in_shape(self, point, shape_data):
        x, y = point;
        shape = shape_data['shape'];
        geom = shape_data['geom']
        if shape == 'circle':
            return (x - geom[0]) ** 2 + (y - geom[1]) ** 2 < geom[2] ** 2
        elif shape == 'rect':
            return geom[0] <= x <= geom[0] + geom[2] and geom[1] <= y <= geom[1] + geom[3]
        elif shape == 'poly':
            return cv2.pointPolygonTest(np.array(geom, dtype=np.int32), (int(x), int(y)), False) >= 0
        return False

    def _get_shape_center(self, shape_data):
        shape = shape_data['shape'];
        geom = shape_data['geom']
        if shape == 'circle':
            return (geom[0], geom[1])
        elif shape == 'rect':
            return (geom[0] + geom[2] // 2, geom[1] + geom[3] // 2)
        elif shape == 'poly':
            pts = np.array(geom)
            return (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
        return (0, 0)