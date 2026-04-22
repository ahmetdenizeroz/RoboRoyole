import cv2
import numpy as np
import os

def extract_even_frames(video_path, output_dir, num_frames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get total frames in this video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames > total_frames:
        print(f"Warning: Requested {num_frames} frames but video only has {total_frames}. Extracting all.")
        num_frames = total_frames

    # Calculate indices evenly from 0 to total_frames - 1
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    print(f"Extracting {num_frames} frames from {total_frames} total frames...")

    count = 0
    for idx in frame_indices:
        # Jump to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Name the file with the frame index to keep it unique
            filename = f"frame_{idx:07d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            count += 1
            
            # Progress print
            if count % 10 == 0 or count == num_frames:
                print(f"Progress: {count}/{num_frames} frames saved.")
        else:
            print(f"Failed to read frame at index {idx}")

    cap.release()
    print(f"Done! Saved {count} frames to {output_dir}")

# Usage
# If you want 300 frames from a 1-hour chunk:
extract_even_frames('bee_chunk_000.mp4', 'dataset_even_spaced', 300)