import cv2
import os
from typing import List, Tuple, Callable, Optional
from pathlib import Path
import re
import matplotlib.pyplot as plt

class ClipExtractorCore:
    def __init__(self):
        pass
        
    @staticmethod
    def parse_coordinates(txt_path: str) -> List[Tuple[int, float, float]]:
        """
        Reads a 3-column tab-separated coordinate file (ID_1_X, ID_1_Y, ID_1_Ang)
        Returns a list of (frame_index, x, y)
        """
        coords = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue # Skip header
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append((idx - 1, x, y))
                    except ValueError:
                        coords.append((idx - 1, -1.0, -1.0))
        return coords

    @staticmethod
    def find_roi_sequences(
        coords: List[Tuple[int, float, float]], 
        roi: Tuple[int, int, int, int], 
        min_frames: int, 
        fps: float,
        pad_seconds: float = 1.0
    ) -> List[Tuple[int, int]]:
        """
        Finds sequences where coordinate is in ROI for at least min_frames.
        ROI is (x, y, w, h).
        Returns a list of (start_frame, end_frame) with padding applied.
        """
        roi_x, roi_y, roi_w, roi_h = roi
        roi_x2, roi_y2 = roi_x + roi_w, roi_y + roi_h
        
        sequences = []
        in_sequence = False
        start_idx = 0
        
        for frame_idx, x, y in coords:
            # Check if point is valid and inside ROI
            is_inside = False
            if x >= 0 and y >= 0:
                if roi_x <= x <= roi_x2 and roi_y <= y <= roi_y2:
                    is_inside = True
                    
            if is_inside:
                if not in_sequence:
                    in_sequence = True
                    start_idx = frame_idx
            else:
                if in_sequence:
                    in_sequence = False
                    length = frame_idx - start_idx
                    if length >= min_frames:
                        sequences.append((start_idx, frame_idx - 1))
                        
        # Check if sequence was still ongoing at the end
        if in_sequence:
            length = coords[-1][0] - start_idx + 1
            if length >= min_frames:
                sequences.append((start_idx, coords[-1][0]))
                
        # Apply padding
        pad_frames = int(round(pad_seconds * fps))
        max_frame = len(coords) - 1
        
        padded_sequences = []
        for start_f, end_f in sequences:
            padded_start = max(0, start_f - pad_frames)
            padded_end = min(max_frame, end_f + pad_frames)
            padded_sequences.append((padded_start, padded_end))
            
        # Merge overlapping sequences after padding
        if not padded_sequences:
            return []
            
        padded_sequences.sort(key=lambda x: x[0])
        merged = [padded_sequences[0]]
        
        for current in padded_sequences[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                # Overlap, merge them
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
                
        return merged

    @staticmethod
    def extract_clips(
        video_path: str, 
        sequences: List[Tuple[int, int]], 
        output_dir: str, 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Extracts video clips for given sequences.
        """
        if not sequences:
            return []
            
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        output_files = []
        base_name = Path(video_path).stem
        
        total_frames_to_process = sum(end_f - start_f + 1 for start_f, end_f in sequences)
        frames_processed = 0
        
        for idx, (start_f, end_f) in enumerate(sequences):
            out_name = f"{base_name}_clip_{idx+1:03d}_{start_f}_to_{end_f}.mp4"
            out_path = os.path.join(output_dir, out_name)
            
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            current_f = start_f
            while current_f <= end_f:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                current_f += 1
                frames_processed += 1
                
                if progress_callback and frames_processed % 10 == 0:
                    progress_callback(frames_processed, total_frames_to_process)
            
            writer.release()
            output_files.append(out_path)
            
        if progress_callback:
            progress_callback(total_frames_to_process, total_frames_to_process)
                
        cap.release()
        return output_files

    @staticmethod
    def parse_info_file(info_path: str) -> dict:
        info = {}
        with open(info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    key = parts[0]
                    val = parts[1]
                    info[key] = val
        return info

    @staticmethod
    def parse_clip_folder(folder_path: str) -> List[Tuple[int, int]]:
        """
        Parses filenames like _clip_001_150_to_300.mp4 and returns list of (start_frame, end_frame)
        """
        sequences = []
        if not folder_path or not os.path.exists(folder_path):
            return sequences
            
        pattern = re.compile(r'_clip_\d+_(\d+)_to_(\d+)\.mp4')
        for filename in os.listdir(folder_path):
            match = pattern.search(filename)
            if match:
                start_f = int(match.group(1))
                end_f = int(match.group(2))
                sequences.append((start_f, end_f))
        
        sequences.sort(key=lambda x: x[0])
        return sequences

    @staticmethod
    def generate_timeline_plot(
        part1_folder: str, part1_info: str,
        part2_folder: str, part2_info: str,
        output_dir: str
    ) -> str:
        
        intervals = []
        total_experiment_s = 0.0
        part1_end_s = 0.0
        
        # Process Part 1
        if part1_info and part1_folder and os.path.exists(part1_info):
            info1 = ClipExtractorCore.parse_info_file(part1_info)
            start_s1 = float(info1.get('start_time_s', 0.0))
            part1_end_s = float(info1.get('end_time_s', 0.0))
            fps1 = float(info1.get('fps', 30.0))
            seqs1 = ClipExtractorCore.parse_clip_folder(part1_folder)
            
            total_experiment_s = part1_end_s
            
            for start_f, end_f in seqs1:
                t_start = start_s1 + (start_f / fps1)
                t_end = start_s1 + (end_f / fps1)
                intervals.append((t_start, t_end))
                
        # Process Part 2 (Optional)
        if part2_info and part2_folder and os.path.exists(part2_info):
            info2 = ClipExtractorCore.parse_info_file(part2_info)
            start_s2 = float(info2.get('start_time_s', 0.0))
            end_s2 = float(info2.get('end_time_s', 0.0))
            fps2 = float(info2.get('fps', 30.0))
            seqs2 = ClipExtractorCore.parse_clip_folder(part2_folder)
            
            total_experiment_s += end_s2
            
            for start_f, end_f in seqs2:
                t_start = start_s2 + (start_f / fps2) + part1_end_s
                t_end = start_s2 + (end_f / fps2) + part1_end_s
                intervals.append((t_start, t_end))
                
        if not intervals:
            raise ValueError("No valid clips found in the provided folders.")
            
        intervals.sort(key=lambda x: x[0])
        
        # Generate plot
        plt.figure(figsize=(12, 3))
        
        for t_start, t_end in intervals:
            plt.hlines(1, t_start / 3600.0, t_end / 3600.0, colors='blue', linewidth=20)
            
        # Add legend for feeding events
        plt.plot([], [], ' ', label=f"Total Feeding Events: {len(intervals)}")
        plt.legend(loc='upper right', frameon=True)
            
        plt.yticks([])
        plt.xlabel("Time (Hours)")
        plt.title("Feeding times")
        
        # Set x limits to show total experiment time
        if total_experiment_s > 0:
            plt.xlim(0, total_experiment_s / 3600.0)
            
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, "feeding_timeline.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        return out_path
