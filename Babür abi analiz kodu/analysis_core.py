# title: "Tracking Data Analysis Core"
# date: "10/27/2025"
# author: "Babur Erdem / Gemini"

import os
import re
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict


# --- Helper Functions ---

def _draw_shape_on_mask(mask, shape_data):
    """Helper to draw a single shape onto a numpy mask."""
    if not shape_data or not shape_data.get('geom'):
        return
    shape = shape_data['shape']
    geom = shape_data['geom']
    try:
        if shape == 'circle':
            cv2.circle(mask, (geom[0], geom[1]), geom[2], 255, -1)
        elif shape == 'rect':
            cv2.rectangle(mask, (geom[0], geom[1]), (geom[0] + geom[2], geom[1] + geom[3]), 255, -1)
        elif shape == 'poly':
            pts = np.array(geom, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    except Exception as e:
        print(f"Warning: Could not draw shape {shape_data.get('id')}: {e}")


def _check_intersection(shape1_data, shape2_data, frame_width, frame_height):
    """Checks if two shape dictionaries intersect using bitwise mask comparison."""
    try:
        mask1 = np.zeros((frame_height, frame_width), dtype=np.uint8)
        mask2 = np.zeros((frame_height, frame_width), dtype=np.uint8)
        _draw_shape_on_mask(mask1, shape1_data)
        _draw_shape_on_mask(mask2, shape2_data)
        intersection = cv2.bitwise_and(mask1, mask2)
        return cv2.countNonZero(intersection) > 0
    except Exception as e:
        print(f"Error checking intersection: {e}")
        return False


def parse_geom(geom_str):
    """Parses geometry string like '[x, y, r]' or '[[x1, y1], [x2, y2], ...]'"""
    try:
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", geom_str)]
        if len(numbers) == 3:  # Circle [cx, cy, r]
            return [int(numbers[0]), int(numbers[1]), int(numbers[2])]
        elif len(numbers) == 4:  # Rectangle [x, y, w, h]
            return [int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])]
        elif len(numbers) > 4 and len(numbers) % 2 == 0:  # Polygon [[x1, y1], ...]
            return [(int(numbers[i]), int(numbers[i + 1])) for i in range(0, len(numbers), 2)]
        else:
            return None
    except Exception as e:
        print(f"Error parsing geometry string '{geom_str}': {e}")
        return None


def get_shape_center(shape_data):
    """Calculates the center point of a shape's geometry dictionary."""
    shape = shape_data['shape']
    geom = shape_data['geom']
    if not geom: return (0, 0)
    if shape == 'circle':
        return (geom[0], geom[1])
    elif shape == 'rect':
        return (geom[0] + geom[2] // 2, geom[1] + geom[3] // 2)
    elif shape == 'poly':
        pts = np.array(geom)
        if len(pts) > 0:
            return (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
        else:
            return (0, 0)
    return (0, 0)


def is_point_in_shape(point, shape_data):
    """Checks if a point (x, y) is inside a shape dictionary."""
    x, y = point
    shape = shape_data['shape']
    geom = shape_data['geom']
    if not geom: return False
    if shape == 'circle':
        if len(geom) != 3: return False
        return (x - geom[0]) ** 2 + (y - geom[1]) ** 2 < geom[2] ** 2
    elif shape == 'rect':
        if len(geom) != 4: return False
        return geom[0] <= x <= geom[0] + geom[2] and geom[1] <= y <= geom[1] + geom[3]
    elif shape == 'poly':
        if not isinstance(geom, list) or len(geom) < 3: return False
        return cv2.pointPolygonTest(np.array(geom, dtype=np.int32), (int(x), int(y)), False) >= 0
    return False


# --- File Parsing ---

def parse_info_file(info_filepath):
    """Reads the _info.txt file and extracts parameters and shape definitions."""
    info = {
        'fps': None, 'pixel_to_mm': 1.0,
        'arenas': {}, 'stimulus_areas': {},
        'frame_width': 2000, 'frame_height': 2000,  # Defaults
        'start_time_str': 'N/A', 'end_time_str': 'N/A'
    }
    current_section = None
    try:
        with open(info_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                if line.startswith('#'):
                    if 'Video Info' in line:
                        current_section = 'video'
                    elif 'Scale' in line:
                        current_section = 'scale'
                    elif 'Arenas' in line:
                        current_section = 'arenas'
                    elif 'Stimulus Areas' in line:
                        current_section = 'stimulus_areas'
                    elif 'id' in line:
                        continue
                    else:
                        current_section = None
                    continue

                parts = line.split('\t')
                if not parts or len(parts) < 2: continue

                key = parts[0]
                value = parts[1]

                if parts[0] == 'id':  # Skip header
                    continue

                if current_section == 'video':
                    if key == 'fps':
                        info['fps'] = float(value)
                    elif key == 'frame_width':
                        info['frame_width'] = int(value)
                    elif key == 'frame_height':
                        info['frame_height'] = int(value)
                    elif key == 'start_time_str':
                        info['start_time_str'] = value
                    elif key == 'end_time_str':
                        info['end_time_str'] = value
                elif current_section == 'scale' and key == 'pixel_to_mm':
                    info['pixel_to_mm'] = float(value)
                elif current_section in ['arenas', 'stimulus_areas'] and len(parts) >= 4:
                    shape_id, shape_type, _, geom_str = parts[0], parts[1], parts[2], parts[3]
                    shape_geom = parse_geom(geom_str)
                    shape_data = {'id': shape_id, 'shape': shape_type, 'geom': shape_geom}
                    shape_data['center'] = get_shape_center(shape_data)

                    if current_section == 'arenas':
                        info['arenas'][shape_id] = shape_data
                    elif current_section == 'stimulus_areas':
                        info['stimulus_areas'][shape_id] = shape_data
    except FileNotFoundError:
        print(f"Error: Info file not found at {info_filepath}")
        return None
    except Exception as e:
        print(f"Error reading info file {info_filepath}: {e}")
        return None
    if info['fps'] is None:
        print(f"Warning: FPS not found in {info_filepath}. Using 30.0 as fallback.")
        info['fps'] = 30.0
    return info


def load_coordinates(coord_filepath):
    """Loads the specified coordinates file into a pandas DataFrame."""
    try:
        df = pd.read_csv(coord_filepath, sep='\t', comment='#')
        df = df.replace(-1, np.nan)
        return df
    except FileNotFoundError:
        print(f"Error: Coordinate file not found at {coord_filepath}")
        return None
    except Exception as e:
        print(f"Error reading coordinate file {coord_filepath}: {e}")
        return None


# --- Intersection Finder ---
def find_intersections(info_data):
    """Finds all Arena/Stimulus intersections and returns a list of tuples."""
    if not info_data:
        return []

    arenas = info_data.get('arenas', {})
    stimuli = info_data.get('stimulus_areas', {})
    w = info_data.get('frame_width')
    h = info_data.get('frame_height')

    intersections = []
    for a_id, arena_info in arenas.items():
        for s_id, stim_info in stimuli.items():
            if _check_intersection(arena_info, stim_info, w, h):
                intersections.append((a_id, s_id))
    return intersections


# --- Time-Binned Analysis ---
def analyze_data_in_bins(experiment_name, coord_df, info_data, bin_seconds, selected_metrics, active_interactions):
    """Performs time-binned analysis based on selected metrics and interactions."""
    if coord_df is None or info_data is None or info_data['fps'] is None or bin_seconds <= 0:
        return None

    fps = info_data['fps']
    pixel_to_mm = info_data['pixel_to_mm']
    arenas = info_data['arenas']
    all_stimulus_areas = info_data['stimulus_areas']

    bin_frames = int(bin_seconds * fps)
    if bin_frames <= 0: return None

    results = []
    animal_ids = sorted(list(set([col.split('_')[1] for col in coord_df.columns])))
    total_frames = len(coord_df)

    for animal_id in animal_ids:
        x_col = f'Animal_{animal_id}_X'
        y_col = f'Animal_{animal_id}_Y'
        if x_col not in coord_df.columns: continue

        animal_coords = coord_df[[x_col, y_col]].rename(columns={x_col: 'x', y_col: 'y'})
        arena_info = arenas.get(animal_id)
        if not arena_info or not arena_info.get('center'):
            arena_center = (np.nan, np.nan)
        else:
            arena_center = arena_info['center']

        relevant_stim_areas = {s_id: all_stimulus_areas[s_id] for (a_id, s_id) in active_interactions if
                               a_id == animal_id}

        for bin_start_frame in range(0, total_frames, bin_frames):
            bin_end_frame = min(bin_start_frame + bin_frames, total_frames)
            if bin_end_frame - bin_start_frame < bin_frames and bin_end_frame < total_frames:
                continue
            time_end_s = bin_end_frame / fps
            bin_data = animal_coords.iloc[bin_start_frame:bin_end_frame].copy()
            valid_data = bin_data.dropna()

            # --- Initialize Metrics ---
            avg_speed_mm_s = np.nan
            avg_dist_arena_center_mm = np.nan
            stim_frames_inside = defaultdict(int)
            stim_entries = defaultdict(int)
            stim_exits = defaultdict(int)
            stim_dist_center_px_sum = defaultdict(float)

            was_in_stim_prev = {stim_id: False for stim_id in relevant_stim_areas}
            if bin_start_frame > 0:
                prev_pos_data = animal_coords.iloc[bin_start_frame - 1]
                if not prev_pos_data.isnull().all():
                    prev_point = (prev_pos_data['x'], prev_pos_data['y'])
                    for stim_id, stim_info in relevant_stim_areas.items():
                        was_in_stim_prev[stim_id] = is_point_in_shape(prev_point, stim_info)

            num_valid_frames = len(valid_data)
            if not valid_data.empty:
                if selected_metrics.get('speed'):
                    diffs = valid_data.diff().dropna()
                    distances = np.sqrt(diffs['x'] ** 2 + diffs['y'] ** 2)
                    total_dist_px = distances.sum()
                    avg_speed_mm_s = (total_dist_px * pixel_to_mm) / bin_seconds if bin_seconds > 0 else 0

                if selected_metrics.get('dist_arena'):
                    dist_to_arena = np.sqrt((valid_data['x'] - arena_center[0]) ** 2 +
                                            (valid_data['y'] - arena_center[1]) ** 2)
                    total_dist_arena_center_px = dist_to_arena.sum()
                    avg_dist_arena_center_mm = (
                                                           total_dist_arena_center_px * pixel_to_mm) / num_valid_frames if num_valid_frames > 0 else np.nan

                # --- CORRECTED: Calculate stim dist for ALL valid frames ---
                if selected_metrics.get('stim_dist'):
                    for stim_id, stim_info in relevant_stim_areas.items():
                        stim_center = stim_info.get('center')
                        if stim_center:
                            dist_to_stim = np.sqrt((valid_data['x'] - stim_center[0]) ** 2 +
                                                   (valid_data['y'] - stim_center[1]) ** 2)
                            stim_dist_center_px_sum[stim_id] = dist_to_stim.sum()

            # --- CORRECTED LOGIC for Entries/Duration/Exits ---
            if selected_metrics.get('stim_duration') or selected_metrics.get('stim_entries'):
                for idx in range(bin_start_frame, bin_end_frame):
                    row = animal_coords.iloc[idx]
                    current_point = (row['x'], row['y'])

                    if pd.isna(current_point[0]):
                        for stim_id in relevant_stim_areas:
                            if was_in_stim_prev[stim_id]:  # If it was in and we lose track, count as exit
                                if selected_metrics.get('stim_entries'):
                                    stim_exits[stim_id] += 1
                            was_in_stim_prev[stim_id] = False
                        continue

                    for stim_id, stim_info in relevant_stim_areas.items():
                        is_in = is_point_in_shape(current_point, stim_info)
                        if is_in:
                            if selected_metrics.get('stim_duration'):
                                stim_frames_inside[stim_id] += 1
                            if selected_metrics.get('stim_entries'):
                                if not was_in_stim_prev[stim_id]:
                                    stim_entries[stim_id] += 1
                        elif was_in_stim_prev[stim_id]:  # Was in, but is now out
                            if selected_metrics.get('stim_entries'):
                                stim_exits[stim_id] += 1

                        was_in_stim_prev[stim_id] = is_in

                        # --- Store Bin Results ---
            bin_result = {
                'ExperimentName': experiment_name,
                'AnimalID': f"{experiment_name}_{animal_id}",
                'Time_End_s': int(time_end_s),
            }
            if selected_metrics.get('speed'):
                bin_result['AvgSpeed_mm_s'] = avg_speed_mm_s
            if selected_metrics.get('dist_arena'):
                bin_result['AvgDistArenaCenter_mm'] = avg_dist_arena_center_mm

            # Store for all stim areas, relevant or not
            for stim_id in all_stimulus_areas:
                if (animal_id, stim_id) in active_interactions:
                    # This stim is relevant, add calculated data
                    if selected_metrics.get('stim_duration'):
                        bin_result[f'Duration_in_Stim_{stim_id}_s'] = stim_frames_inside[stim_id] / fps
                    if selected_metrics.get('stim_dist'):
                        avg_dist_stim_center_mm = (stim_dist_center_px_sum[
                                                       stim_id] * pixel_to_mm) / num_valid_frames if num_valid_frames > 0 else np.nan
                        bin_result[f'AvgDistStimCenter_{stim_id}_mm'] = avg_dist_stim_center_mm
                    if selected_metrics.get('stim_entries'):
                        bin_result[f'Entries_to_Stim_{stim_id}'] = stim_entries[stim_id]
                        bin_result[f'Exits_from_Stim_{stim_id}'] = stim_exits[stim_id]
                else:
                    # Add NaN (empty cells) for non-active interactions
                    if selected_metrics.get('stim_duration'): bin_result[f'Duration_in_Stim_{stim_id}_s'] = np.nan
                    if selected_metrics.get('stim_dist'): bin_result[f'AvgDistStimCenter_{stim_id}_mm'] = np.nan
                    if selected_metrics.get('stim_entries'):
                        bin_result[f'Entries_to_Stim_{stim_id}'] = np.nan
                        bin_result[f'Exits_from_Stim_{stim_id}'] = np.nan

            results.append(bin_result)

    return pd.DataFrame(results)


# --- Visualization Functions ---

def create_heatmaps(coord_df, info_data, save_directory, experiment_name):
    """Creates and saves a heatmap for each animal."""
    if coord_df is None or info_data is None:
        print("No data for heatmaps.")
        return

    animal_ids = sorted(list(set([col.split('_')[1] for col in coord_df.columns])))
    arenas = info_data.get('arenas', {})
    w = info_data.get('frame_width')
    h = info_data.get('frame_height')

    for animal_id in animal_ids:
        animal_id_full = f"{experiment_name}_{animal_id}"  # Use new ID format
        x_col = f'Animal_{animal_id}_X'
        y_col = f'Animal_{animal_id}_Y'
        if x_col not in coord_df.columns: continue

        animal_coords = coord_df[[x_col, y_col]].dropna()
        if animal_coords.empty:
            print(f"No valid coordinates for {animal_id} to generate heatmap.")
            continue

        x = animal_coords[x_col].to_numpy()
        y = animal_coords[y_col].to_numpy()

        # Create 2D histogram
        # Use a fixed bin size for better resolution, e.g., 20px bins
        bins_x = int(w / 20)
        bins_y = int(h / 20)
        range_hist = [[0, w], [0, h]]

        hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y], range=range_hist)
        hist = hist.T  # Transpose to match (x, y) orientation

        # Process for visualization
        hist = cv2.GaussianBlur(hist, (11, 11), 0)  # Smooth the histogram more
        hist = np.sqrt(hist)  # Use sqrt to better show low-density areas
        hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

        # Create a mask of the arena
        arena_mask = np.zeros((bins_y, bins_x), dtype=np.uint8)  # Match histogram dimensions
        arena_info = arenas.get(animal_id)

        if arena_info:
            scaled_geom = {}
            shape = arena_info['shape']
            geom = arena_info['geom']
            scaled_geom['shape'] = shape

            # Scale geometry to the histogram's bin dimensions
            scale_x = bins_x / w
            scale_y = bins_y / h

            if shape == 'circle':
                scaled_geom['geom'] = [int(geom[0] * scale_x), int(geom[1] * scale_y), int(geom[2] * scale_x)]
            elif shape == 'rect':
                scaled_geom['geom'] = [int(geom[0] * scale_x), int(geom[1] * scale_y), int(geom[2] * scale_x),
                                       int(geom[3] * scale_y)]
            elif shape == 'poly':
                scaled_geom['geom'] = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in geom]

            _draw_shape_on_mask(arena_mask, scaled_geom)
            # Apply mask
            heatmap = cv2.bitwise_and(heatmap, heatmap, mask=arena_mask)

        save_path = os.path.join(save_directory, f"{animal_id_full}_heatmap.png")
        try:
            cv2.imwrite(save_path, heatmap)
            print(f"Saved heatmap to {save_path}")
        except Exception as e:
            print(f"Error saving heatmap {save_path}: {e}")


def create_visualizations(results_df, info_data, save_directory, experiment_name, selected_metrics):
    """Creates and saves time-series plots for each animal based on selected metrics."""
    if results_df is None or results_df.empty:
        print("No data to plot.")
        return

    animal_ids = results_df['AnimalID'].unique()  # These are already ExperimentName_A1
    stimulus_ids = sorted(list(info_data['stimulus_areas'].keys()))

    for animal_id in animal_ids:
        animal_df = results_df[results_df['AnimalID'] == animal_id].copy()
        if animal_df.empty:
            continue

        time_col = 'Time_End_s'

        # Determine number of plots
        plot_list = []
        if selected_metrics.get('speed'): plot_list.append('speed')
        if selected_metrics.get('dist_arena'): plot_list.append('dist_arena')

        relevant_stim_cols = []
        for s_id in stimulus_ids:
            if f'Duration_in_Stim_{s_id}_s' in animal_df.columns:
                if animal_df[f'Duration_in_Stim_{s_id}_s'].notna().any():
                    relevant_stim_cols.append(s_id)

        if selected_metrics.get('stim_duration'):
            for s_id in relevant_stim_cols: plot_list.append(f'dur_{s_id}')
        if selected_metrics.get('stim_dist'):
            for s_id in relevant_stim_cols: plot_list.append(f'dist_stim_{s_id}')
        if selected_metrics.get('stim_entries'):
            for s_id in relevant_stim_cols:
                plot_list.append(f'ent_{s_id}')
                plot_list.append(f'exi_{s_id}')  # Add exits plot

        num_rows = len(plot_list)
        if num_rows == 0:
            print(f"No metrics selected to plot for {animal_id}.")
            continue

        fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 3 * num_rows), sharex=True)
        if num_rows == 1:
            axes = [axes]

        fig.suptitle(f"Time-Binned Analysis: {animal_id}", fontsize=16)

        plot_idx = 0
        for plot_type in plot_list:
            ax = axes[plot_idx]
            if plot_type == 'speed':
                ax.plot(animal_df[time_col], animal_df['AvgSpeed_mm_s'], marker='o', linestyle='-')
                ax.set_ylabel("Avg Speed (mm/s)")
                ax.set_title("Average Speed per Time Bin")
                ax.set_ylim(bottom=0)
            elif plot_type == 'dist_arena':
                ax.plot(animal_df[time_col], animal_df['AvgDistArenaCenter_mm'], marker='o', linestyle='-', color='g')
                ax.set_ylabel("Avg Distance (mm)")
                ax.set_title("Average Distance to Arena Center per Time Bin")
                ax.set_ylim(bottom=0)
            elif plot_type.startswith('dur_'):
                s_id = plot_type.split('_')[1]
                ax.plot(animal_df[time_col], animal_df[f'Duration_in_Stim_{s_id}_s'], marker='s', linestyle='-',
                        color='r')
                ax.set_ylabel("Duration (s)")
                ax.set_title(f"Duration in Stimulus Area {s_id} per Bin")
                ax.set_ylim(bottom=0)
            elif plot_type.startswith('dist_stim_'):
                s_id = plot_type.split('_')[2]
                ax.plot(animal_df[time_col], animal_df[f'AvgDistStimCenter_{s_id}_mm'], marker='P', linestyle='-',
                        color='m')
                ax.set_ylabel("Avg Distance (mm)")
                ax.set_title(f"Avg Distance to Stimulus {s_id} Center (All Frames)")
                ax.set_ylim(bottom=0)
            elif plot_type.startswith('ent_'):
                s_id = plot_type.split('_')[1]
                ax.plot(animal_df[time_col], animal_df[f'Entries_to_Stim_{s_id}'], marker='x', linestyle='-', color='c')
                ax.set_ylabel("Count")
                ax.set_title(f"Entries to Stimulus Area {s_id} per Bin")
                ax.set_ylim(bottom=0)
            elif plot_type.startswith('exi_'):
                s_id = plot_type.split('_')[1]
                ax.plot(animal_df[time_col], animal_df[f'Exits_from_Stim_{s_id}'], marker='x', linestyle='-',
                        color='orange')
                ax.set_ylabel("Count")
                ax.set_title(f"Exits from Stimulus Area {s_id} per Bin")
                ax.set_ylim(bottom=0)

            ax.grid(True, linestyle=':')
            plot_idx += 1

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        save_path = os.path.join(save_directory, f"{animal_id}_analysis_plots.png")
        try:
            fig.savefig(save_path)
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Error saving plot {save_path}: {e}")

        plt.close(fig)
    return True