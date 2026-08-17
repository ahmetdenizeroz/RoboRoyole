import cv2
import numpy as np

class VideoProcessorCore:
    def __init__(self):
        self.transforms_smooth = []
        self.is_calculated = False

    def moving_average(self, curve, radius):
        window_size = 2 * radius + 1
        f = np.ones(window_size) / window_size
        curve_pad = np.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        curve_smoothed = curve_smoothed[radius:-radius]
        return curve_smoothed

    def smooth_trajectory(self, trajectory, smoothing_radius=50):
        smoothed = np.copy(trajectory)
        for i in range(3):
            smoothed[:, i] = self.moving_average(trajectory[:, i], radius=smoothing_radius)
        return smoothed

    def calculate_stabilization(self, video_path: str, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 1:
            cap.release()
            return
            
        ret, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        
        transforms = []
        
        # We need N-1 transforms
        for i in range(n_frames - 1):
            ret, curr = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            # Find feature points
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            
            if prev_pts is None:
                # If no features, assume no movement
                transforms.append([0.0, 0.0, 0.0])
                prev_gray = curr_gray
                if progress_callback:
                    progress_callback(int((i / (n_frames - 1)) * 100))
                continue
                
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            
            # Filter good points
            idx = np.where(status == 1)[0]
            if len(idx) == 0:
                transforms.append([0.0, 0.0, 0.0])
            else:
                prev_pts_good = prev_pts[idx]
                curr_pts_good = curr_pts[idx]
                
                # Estimate affine transform
                m, inliers = cv2.estimateAffinePartial2D(prev_pts_good, curr_pts_good)
                if m is None:
                    transforms.append([0.0, 0.0, 0.0])
                else:
                    dx = m[0, 2]
                    dy = m[1, 2]
                    da = np.arctan2(m[1, 0], m[0, 0])
                    transforms.append([dx, dy, da])
            
            prev_gray = curr_gray
            if progress_callback:
                progress_callback(int((i / (n_frames - 1)) * 100))
                
        cap.release()
        
        if not transforms:
            return
            
        transforms = np.array(transforms)
        
        # Compute trajectory (cumulative sum of transforms)
        trajectory = np.cumsum(transforms, axis=0)
        
        # Smooth trajectory
        smoothed_trajectory = self.smooth_trajectory(trajectory, smoothing_radius=50)
        
        # Calculate new transforms
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference
        
        # Frame 0 gets no transform, subsequent frames get transforms_smooth
        self.transforms_smooth = np.vstack([[0.0, 0.0, 0.0], transforms_smooth])
        self.is_calculated = True

    def apply_filters(self, frame: np.ndarray, frame_idx: int, settings: dict) -> np.ndarray:
        if frame is None:
            return None
            
        # 1. Stabilization
        if settings.get("use_stabilization", False) and self.is_calculated:
            # frame_idx might exceed if video is wonky, safely cap it
            safe_idx = min(frame_idx, len(self.transforms_smooth) - 1)
            dx, dy, da = self.transforms_smooth[safe_idx]
            
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy
            
            h, w = frame.shape[:2]
            frame = cv2.warpAffine(frame, m, (w, h))
            
        # 2. Brightness & Contrast
        alpha = settings.get("contrast", 1.0)
        beta = settings.get("brightness", 0)
        if alpha != 1.0 or beta != 0:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
        # 3. CLAHE
        if settings.get("use_clahe", False):
            is_color = len(frame.shape) == 3
            if is_color:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            clip_limit = settings.get("clahe_clip", 2.0)
            grid_size = settings.get("clahe_grid", 8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            filtered = clahe.apply(gray)
            
            frame = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            
        # 4. Morphology Top-Hat / Black-Hat
        if settings.get("use_morphology", False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            k_size = settings.get("morph_size", 15)
            # Make sure kernel size is odd
            if k_size % 2 == 0:
                k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            
            morph_type = settings.get("morph_type", "Top-Hat")
            if morph_type == "Black-Hat":
                filtered = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            else:
                filtered = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
                
            frame = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            
        return frame
