#For your research article: It is zero problem. You do not need special permission. All you need to do is standard academic practice: cite Tesseract in your paper's methodology section (e.g., "Optical Character Recognition was performed using the Tesseract OCR engine (Smith, 2007).")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import csv
import re
import os
import subprocess
import cv2
import pytesseract
from scipy.interpolate import PchipInterpolator
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Definitions
tags = [0]

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_DIR = r'C:\Program Files\Tesseract-OCR\tessdata'

# Add this magic bullet line:
# This forces Tesseract to completely ignore your old Windows settings
os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

CROP_Y1, CROP_Y2 = 650, 720
CROP_X1, CROP_X2 = 1100, 1280

# --- LOG PARSING SETTINGS ---
# The current log format is like:
# [2026-05-04 12:00:15] [Arduino]: SENSORS: B=ON 535 M=OFF 314 F=OFF 235 | Pattern: WAITING_TARGET | State: WAITING
# These regexes also support the older format without the Pattern field:
# ... SENSORS: B=ON 535 M=OFF 314 F=OFF 235 | State: WAITING
F_THRESHOLD = 300
TIME_PATTERN = re.compile(r'^\[([\d\-\s:]+)\]')
F_VALUE_PATTERN = re.compile(r'F=(?:ON|OFF)\s+(\d+)')
SENSOR_LINE_PATTERN = re.compile(
    r"SENSORS:\s*"
    r"B=(?:ON|OFF)\s+(\d+)\s+"
    r"M=(?:ON|OFF)\s+(\d+)\s+"
    r"F=(?:ON|OFF)\s+(\d+)"
    r"(?:\s+\|\s+Pattern:\s+[^|]+)?"
    r"\s+\|\s+State:\s+(\w+)"
)
STATE_PATTERN = re.compile(r'\|\s*State:\s*(\w+)')

def validate_aruco_tags(file_path, expected_tags):
    """
    Compares detected Aruco IDs in the log against a list of expected IDs.
    
    Args:
        file_path (str): Path to the log file.
        expected_tags (list): List of integers or strings of tags you put on the bees.
    """
    # Convert all expected tags to strings for consistent comparison
    expected_set = set(str(t) for t in expected_tags)
    detected_set = set()
    
    tag_pattern = re.compile(r"Tag (\d+) confirmed")

    if not os.path.exists(file_path):
        print(f"Error: Log file {file_path} not found.")
        return

    # 1. Scan the log for all unique confirmed tags
    with open(file_path, 'r') as file:
        for line in file:
            if "confirmed. Sending FEED command." in line:
                match = tag_pattern.search(line)
                if match:
                    detected_set.add(match.group(1))

    # 2. Perform the Comparison
    ghost_tags = detected_set - expected_set  # In log, but NOT in your list
    missing_tags = expected_set - detected_set # In your list, but NOT in log

    # 3. Output Results
    print("\n" + "="*40)
    print("      ARUCO TAG VALIDATION REPORT")
    print("="*40)

    print(f"Total Expected: {len(expected_set)}")
    print(f"Total Detected: {len(detected_set)}")
    print("-" * 40)

    if not ghost_tags and not missing_tags:
        print("✅ SUCCESS: All expected tags were detected and no ghost tags found.")
    else:
        if ghost_tags:
            print(f"❌ GHOST TAGS (Detected but not expected):")
            for tag in sorted(ghost_tags, key=int):
                print(f"   - ID: {tag}")
        else:
            print("✅ No Ghost Tags detected.")

        print("")

        if missing_tags:
            print(f"❌ MISSING TAGS (Expected but never detected):")
            for tag in sorted(missing_tags, key=int):
                print(f"   - ID: {tag}")
        else:
            print("✅ All expected bees were detected at least once.")
    
    print("="*40 + "\n")

    return list(detected_set)

def calculate_total_log_time(file_path):
    # 1. Initialize our start and end variables
    start_time = None
    end_time = None

    # 2. Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            
            # Fast filter
            if not clean_line.startswith('['):
                continue
                
            time_str = clean_line[1:20]
            try:
                # Convert text to datetime
                current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                
                # 3. Capture the very first time
                if start_time is None:
                    start_time = current_time
                
                # 4. Continuously overwrite end_time so it holds the last logged time
                end_time = current_time
                
            except ValueError:
                continue

    # 5. Calculate the total duration
    if start_time and end_time:
        # Get the difference between the two times
        time_difference = end_time - start_time
        
        # Get the absolute total in seconds
        total_seconds = int(time_difference.total_seconds())
        
        # 6. Math magic to break seconds down into Hours, Minutes, and Seconds
        hours = total_seconds // 3600          # Integer division to get total full hours
        minutes = (total_seconds % 3600) // 60 # Remainder of hours divided by 60 for minutes
        seconds = total_seconds % 60           # Remainder of minutes for the leftover seconds
        
        print("--- Log Duration Summary ---")
        print(f"Log Started:  {start_time}")
        print(f"Log Ended:    {end_time}")
        print("-" * 26)
        print(f"Total Uptime: {hours} hours, {minutes} minutes, {seconds} seconds")
    else:
        print("Could not find valid timestamps in the log file.")

def number_of_lines(file_path, target_tags):
    # 1. Initialize our counters
    total_lines = 0
    # Create a dictionary to count feeds for each specific bee
    # e.g., {'667': 0, '183': 0}
    tag_counts = {str(tag): 0 for tag in target_tags}
    
    state_counts = {
        'IDLE': 0,
        'HOLDING': 0,
        'WAITING': 0,
        'EJECTING': 0,
        'RETRACTING': 0
    }
    
    # 2. Open and read the file
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as file:
        for line in file:
            total_lines += 1
            clean_line = line.strip()
            
            # 3. Check for specific Bee Tag triggers
            if "confirmed. Sending FEED command." in clean_line:
                for tag in tag_counts.keys():
                    if f"Tag {tag} confirmed" in clean_line:
                        tag_counts[tag] += 1
                        break # Found the bee, no need to check other tags for this line
                
            # 4. Check for state information
            # Supports both:
            #   ... | State: HOLDING
            #   ... | Pattern: FULL | State: HOLDING
            state_match = STATE_PATTERN.search(clean_line)
            if state_match:
                current_state = state_match.group(1)
                if current_state in state_counts:
                    state_counts[current_state] += 1

    # 5. Print the multi-bee results
    print(f"\n--- Multi-Bee Log Analysis ---")
    print(f"Total Lines Processed: {total_lines}")
    print("-" * 30)
    print("Feed Commands per Bee:")
    for tag, count in tag_counts.items():
        print(f"  Bee {tag}: {count} feeds")
    
    print("-" * 30)
    print("System State Totals:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} lines")
    print("-" * 30 + "\n")

def feeding_trigger_vs_time(file_path, target_tags):
    """
    Generates activity plots for multiple bee tags.
    target_tags: list of integers or strings (e.g., [667, 183])
    """
    # 1. Prepare data structures for each tag
    # We use a dictionary where the key is the tag and the value is the list of feed_values
    tag_data = {str(tag): [] for tag in target_tags}
    times = []
    
    # 2. Open and read the file once
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            if not clean_line.startswith('['):
                continue
                
            try:
                time_str = clean_line[1:20]
                current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                times.append(current_time)
                
                # Check each tag individually for this specific line
                for tag in tag_data.keys():
                    phrase = f"Tag {tag} confirmed. Sending FEED command."
                    if phrase in clean_line:
                        tag_data[tag].append(1)
                    else:
                        tag_data[tag].append(0)
                        
            except ValueError:
                continue

    if not times:
        print("No valid timestamps found to plot.")
        return

    # 3. Generate Subplots
    num_tags = len(target_tags)
    fig, axes = plt.subplots(num_tags, 1, figsize=(14, 3 * num_tags), sharex=True)
    
    # If there is only one tag, 'axes' isn't a list, so we wrap it
    if num_tags == 1:
        axes = [axes]

    print(f"--- Multi-Tag Activity Summary ---")
    for i, tag in enumerate(tag_data.keys()):
        ax = axes[i]
        values = tag_data[tag]
        total_feeds = sum(values)
        
        print(f"Tag {tag}: {total_feeds} feed commands.")

        # Plotting
        ax.plot(times, values, marker='.', linestyle='-', color=plt.cm.tab10(i % 10), 
                markersize=2, alpha=0.7, label=f'Bee {tag}')
        
        # Formatting
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Other', 'FEED'])
        ax.set_ylabel(f'Bee {tag}')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    # Global Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    axes[0].set_title('Feeding Activity Comparison Across Multiple Tags', pad=20, fontweight='bold')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

def time_data_for_feeding(input_file, output_file, target_tags):
    # Convert tags to strings for comparison
    target_tags = [str(t) for t in target_tags]

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Added 'Tag ID' as the first column
        writer.writerow(['Tag ID', 'Tag Time', 'Eject Time (s)', 'Hold Time (s)', 'Retract Time (s)', 'Detected During Retract'])
        
        in_sequence = False
        current_phase = "IDLE" 
        
        current_tag_id = ""   # Tracks which bee 'owns' the current sequence
        tag_time_str = ""
        is_retract_tag = False 
        
        eject_start = eject_end = None
        hold_start = hold_end = None
        retract_start = retract_end = None
        
        sequence_count = 0

        for line in infile:
            clean_line = line.strip()
            if not clean_line.startswith('['): continue
                
            time_str = clean_line[1:20]
            try:
                current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError: continue

            # --- 1. DYNAMIC TAG DETECTION ---
            # We look for ANY of the provided tags in the 'confirmed' line
            if "confirmed. Sending FEED command." in clean_line:
                detected_tag = None
                for t in target_tags:
                    if f"Tag {t} confirmed" in clean_line:
                        detected_tag = t
                        break
                
                # If the tag is in our target list and we are in a position to start/restart
                if detected_tag and current_phase in ["IDLE", "RETRACTING"]:
                    
                    # If it's an interruption during retraction, save the previous bee's data
                    if current_phase == "RETRACTING":
                        write_sequence_to_csv(writer, current_tag_id, tag_time_str, eject_start, eject_end, 
                                            hold_start, hold_end, retract_start, current_time, is_retract_tag)
                        sequence_count += 1
                        is_retract_tag = True # This new sequence was triggered during a retract
                    else:
                        is_retract_tag = False
                    
                    # Start New Sequence for the NEW bee
                    in_sequence = True
                    current_tag_id = detected_tag
                    tag_time_str = time_str
                    eject_start = current_time
                    eject_end = hold_start = hold_end = retract_start = retract_end = None
                    current_phase = "EJECTING"
                
                continue

            # --- 2. PHASE TRANSITIONS (The Anchors) ---
            if "Starting HOLD" in clean_line and in_sequence:
                eject_end = current_time
                hold_start = current_time
                current_phase = "HOLDING"

            elif "Hold time over. Retracting." in clean_line and in_sequence:
                hold_end = current_time
                retract_start = current_time
                current_phase = "RETRACTING"

            elif "Retract Complete" in clean_line and in_sequence:
                retract_end = current_time
                write_sequence_to_csv(writer, current_tag_id, tag_time_str, eject_start, eject_end, 
                                    hold_start, hold_end, retract_start, retract_end, is_retract_tag)
                sequence_count += 1
                
                # Reset to IDLE
                in_sequence = False
                current_phase = "IDLE"
                current_tag_id = ""
                tag_time_str = ""

    print(f"Extraction complete! {sequence_count} sequences for {len(target_tags)} bees saved to {output_file}.")

def write_sequence_to_csv(writer, tag_id, tag_time, e_start, e_end, h_start, h_end, r_start, r_end, is_retract):
    e_time = (e_end - e_start).total_seconds() if (e_start and e_end) else 0
    h_time = (h_end - h_start).total_seconds() if (h_start and h_end) else 0
    r_time = (r_end - r_start).total_seconds() if (r_start and r_end) else 0
    
    retract_str = "Yes" if is_retract else "No"
    # Added tag_id to the writer row
    writer.writerow([tag_id, tag_time, e_time, h_time, r_time, retract_str])

# Possible feeding analysis
def analyze_possible_feeding(file_path, output_csv):
    """
    Groups consecutive secondary feeding detections into 'chains'.
    Calculates min, max, avg intensity, and duration for each chain.
    Saves to CSV and plots the start of each chain.
    """
    times_for_plot = []
    binary_values = []
    chain_data_for_csv = []
    
    target_phrase = "Possible feeding detected at secondary zone:"
    
    # Temporary storage for the current active chain
    current_chain_times = []  # Stores datetime objects
    current_chain_values = [] # Stores float intensities
    last_processed_time = None

    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            if not clean_line.startswith('['):
                continue
                
            try:
                # Extract timestamp
                time_str = clean_line[1:20]
                current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                
                # Build the global timeline for the plot (initially all 0)
                times_for_plot.append(current_time)
                binary_values.append(0)
                
                if target_phrase in clean_line:
                    # Extract numeric value
                    val_str = clean_line.split(":")[-1].strip()
                    val = float(val_str)
                    
                    # Logic: Is this within 1 second of the last detection?
                    if last_processed_time and (current_time - last_processed_time).total_seconds() <= 1.0:
                        current_chain_times.append(current_time)
                        current_chain_values.append(val)
                    else:
                        # Before starting a new chain, save the previous one if it exists
                        if current_chain_values:
                            start_dt = current_chain_times[0]
                            end_dt = current_chain_times[-1]
                            duration = (end_dt - start_dt).total_seconds()
                            
                            mn, mx = min(current_chain_values), max(current_chain_values)
                            avg = sum(current_chain_values) / len(current_chain_values)
                            
                            chain_data_for_csv.append([
                                start_dt.strftime("%Y-%m-%d %H:%M:%S"), 
                                mn, mx, round(avg, 2), len(current_chain_values), duration
                            ])
                            
                            # Mark the START of the chain in the binary plot
                            # We find the first occurrence of this specific second in our plot list
                            for idx, t in enumerate(times_for_plot):
                                if t == start_dt:
                                    binary_values[idx] = 1
                                    break

                        # Start a fresh chain
                        current_chain_times = [current_time]
                        current_chain_values = [val]
                    
                    last_processed_time = current_time
                    
            except (ValueError, IndexError):
                continue

        # Finalize the very last chain in the file
        if current_chain_values:
            start_dt = current_chain_times[0]
            end_dt = current_chain_times[-1]
            duration = (end_dt - start_dt).total_seconds()
            mn, mx = min(current_chain_values), max(current_chain_values)
            avg = sum(current_chain_values) / len(current_chain_values)
            
            chain_data_for_csv.append([
                start_dt.strftime("%Y-%m-%d %H:%M:%S"), 
                mn, mx, round(avg, 2), len(current_chain_values), duration
            ])
            for idx, t in enumerate(times_for_plot):
                if t == start_dt:
                    binary_values[idx] = 1
                    break

    # 1. Save Chains to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Start Timestamp', 'Min Intensity', 'Max Intensity', 'Avg Intensity', 'Log Count', 'Duration (s)'])
        writer.writerows(chain_data_for_csv)

    print(f"Analysis complete. Found {len(chain_data_for_csv)} distinct feeding chains.")

    # 2. Plotting
    if not times_for_plot:
        print("No data to plot.")
        return

    plt.figure(figsize=(15, 6))
    plt.plot(times_for_plot, binary_values, color='teal', drawstyle='steps-pre', linewidth=1.5)
    
    # Formatting
    plt.title('Secondary Feeding Events: Chain Starts Only', fontsize=14, fontweight='bold')
    plt.xlabel('Time (HH:MM:SS)')
    plt.ylabel('Event Trigger')
    plt.yticks([0, 1], ['Idle', 'CHAIN START'])
    plt.ylim(-0.2, 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

# front sensor analysis 
def plot_f_sensor_over_time(file_path, duration_str=None):
    # 1. Parse the optional duration limit
    time_limit = None
    if duration_str:
        try:
            h, m, s = map(int, duration_str.split(':'))
            time_limit = timedelta(hours=h, minutes=m, seconds=s)
            print(f"Limiting analysis to the first {duration_str} (HH:MM:SS) of the log.")
        except ValueError:
            print("Invalid duration format. Please use 'HH:MM:SS'. Proceeding with full log.")

    # Initialize our lists and start time tracker
    times = []
    f_values = []
    start_time = None

    # 2. Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            
            # Fast filter: Skip lines that don't have timestamps or sensor data
            if not clean_line.startswith('[') or "SENSORS:" not in clean_line:
                continue
                
            # Extract the time
            time_str = clean_line[1:20]
            try:
                current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            # --- THE TIME LIMIT CHECK ---
            # Capture the very first valid timestamp to anchor our timeline
            if start_time is None:
                start_time = current_time
            
            # If a limit was provided, check if we've passed it
            if time_limit is not None and (current_time - start_time) > time_limit:
                print("Reached the requested time limit. Stopping log extraction.")
                break # This instantly stops the loop and moves to plotting!
            # ----------------------------

            # Extract the F sensor value using regex
            f_match = F_VALUE_PATTERN.search(clean_line)
            
            if f_match:
                # If we got both a valid time and a valid sensor reading, save them
                times.append(current_time)
                f_values.append(int(f_match.group(1)))

    print(f"Extracted {len(times)} data points for the F sensor.")

    # 3. Generate the Graph
    if times:
        plt.figure(figsize=(14, 6)) # A nice wide canvas for a time series
        
        # Plot the data
        plt.plot(times, f_values, marker='.', linestyle='-', color='teal', markersize=2, linewidth=0.5, alpha=0.8)
        
        # Format the X-axis (Time)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate() # Rotate dates slightly so they don't overlap
        
        # Add labels, title, and styling
        plt.title('Front (F) Sensor Values Over Time', pad=15, fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('F Sensor Raw Value', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Show the plot
        plt.show()
    else:
        print("No F sensor data found to plot.")

def plot_f_sensor_binary(file_path, duration_str=None):
    # 1. Parse Duration
    time_limit = None
    if duration_str:
        try:
            h, m, s = map(int, duration_str.split(':'))
            time_limit = timedelta(hours=h, minutes=m, seconds=s)
        except ValueError:
            print("Invalid duration format. Using full log.")

    times = []
    f_binary_values = []
    start_time = None

    # 2. Shared Regex Patterns
    # Matches the timestamp: [2026-03-23 18:41:55]
    time_pattern = TIME_PATTERN
    
    # Matches examples such as F=OFF 11 or F=ON 440
    f_pattern = F_VALUE_PATTERN

    with open(file_path, 'r') as file:
        for line in file:
            # We only care about the SENSORS lines
            if "SENSORS:" not in line:
                continue
            
            # 3. Extract Time
            time_match = time_pattern.search(line)
            if not time_match:
                continue
            
            try:
                current_time = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            if start_time is None:
                start_time = current_time
            
            if time_limit and (current_time - start_time) > time_limit:
                break 

            # 4. Extract F Sensor Value
            f_match = f_pattern.search(line)
            if f_match:
                f_raw_value = int(f_match.group(1))
                # Binary threshold: 1 if above F_THRESHOLD, else 0
                f_binary_values.append(1 if f_raw_value > F_THRESHOLD else 0)
                times.append(current_time)

    # 5. Generate the Plot
    if times:
        plt.figure(figsize=(14, 5))
        
        # Step plot works great for binary data to show distinct "On/Off" states
        plt.step(times, f_binary_values, where='post', color='purple', linewidth=1.5)
        plt.fill_between(times, f_binary_values, step="post", alpha=0.2, color='purple')
        
        # Formatting
        plt.ylim(-0.2, 1.2)
        plt.yticks([0, 1], [f'Normal (<= {F_THRESHOLD})', f'Triggered (> {F_THRESHOLD})']) 
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        
        plt.title('Front (F) Sensor Activity Threshold Tracking', pad=15, fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No F sensor data found in the provided log file.")

# Video tools
def generate_drift_samples(video_file, total_video_duration_str, num_samples=20, debug=True):
    temp_dir = "ocr_debug_frames"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 1. Calculate Target Video Times
    h, m, s = map(int, total_video_duration_str.split(':'))
    total_video_sec = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
    
    # Create evenly spaced intervals
    interval = total_video_sec / num_samples
    target_video_seconds = [interval * i for i in range(num_samples)]
    target_video_seconds.append(total_video_sec - 5) # Ensure last sample is near the end

    results = []
    days_passed = 0
    prev_ocr_sec = None

    print(f"Starting pipeline. Extracting {len(target_video_seconds)} samples...")

    # 2. Process Each Target Time
    for idx, vid_sec in enumerate(target_video_seconds):
        success, ocr_time_str = extract_and_read_frame(video_file, vid_sec, temp_dir, idx, debug)
        
        # Retry logic: If OCR fails, step forward 2 seconds and try again
        if not success:
            print(f"Sample {idx}: OCR failed or unreadable. Retrying +2 seconds...")
            vid_sec += 2
            success, ocr_time_str = extract_and_read_frame(video_file, vid_sec, temp_dir, idx, debug)

        if success:
            # 3. Handle Time Math and Day Rollovers
            h, m, s = map(int, ocr_time_str.split(':'))
            current_ocr_sec_of_day = (h * 3600) + (m * 60) + s
            
            # Day Rollover Logic
            if prev_ocr_sec is not None:
                # If the new time is significantly smaller than the previous time, we crossed midnight
                if current_ocr_sec_of_day < (prev_ocr_sec - 43200): # 12-hour buffer
                    days_passed += 1
            
            absolute_ocr_sec = (days_passed * 86400) + current_ocr_sec_of_day
            prev_ocr_sec = current_ocr_sec_of_day
            
            # Format nicely for the output
            vid_time_formatted = str(timedelta(seconds=int(vid_sec)))
            results.append((vid_time_formatted, ocr_time_str, vid_sec, absolute_ocr_sec))
            print(f"Sample {idx:02d} | Video: {vid_time_formatted} -> Stamp: {ocr_time_str}")
        else:
            print(f"Sample {idx:02d} | Video: {str(timedelta(seconds=int(vid_sec)))} -> FAILED TO READ")

    # 4. Print the final usable map
    print("\n--- Final Drift Mapping ---")
    for res in results:
        print(f"Video Time: {res[0]} = Log Time: {res[1]}")
    
    return results

def extract_and_read_frame(video_file, time_in_seconds, temp_dir, idx, debug):
    raw_frame_path = os.path.join(temp_dir, f"raw_{idx}.jpg")
    debug_crop_path = os.path.join(temp_dir, f"crop_{idx}.jpg")

    # 1. FFmpeg extraction (Fast Seek)
    cmd = [
        'ffmpeg', '-y', '-ss', str(time_in_seconds), 
        '-i', video_file, '-vframes', '1', '-q:v', '2', raw_frame_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    if not os.path.exists(raw_frame_path):
        return False, None

    # 2. OpenCV Image Processing
    img = cv2.imread(raw_frame_path)
    if img is None: return False, None

    # Crop to the bottom right
    roi = img[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Thresholding: Push the bright white text to pure white (255) and the red background to black (0)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    if debug:
        cv2.imwrite(debug_crop_path, thresh)

    # 3. Tesseract OCR
    # Nice and clean, just using the deep learning engine and the whitelist
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    text = pytesseract.image_to_string(thresh, config=config).strip()

    # Clean up the raw full-size frame to save disk space
    os.remove(raw_frame_path)

    # 4. Regex Validation
    # We strictly enforce HH:MM:SS format
    match = re.search(r'^([0-1]?[0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])$', text)
    if match:
        return True, match.group(0)
    else:
        return False, text

def build_drift_model(drift_data, debug=True):
    if not drift_data:
        raise ValueError("No valid drift samples collected.")

    print("\nFiltering outliers and building PCHIP model...")
    
    filtered_x = []
    filtered_y = []
    first_ocr_sec = drift_data[0][3]
    
    # We'll use these to track the last 'known good' point
    last_good_x = 0.0
    last_good_y = drift_data[0][2] 

    for i, row in enumerate(drift_data):
        vid_sec = float(row[2])
        abs_ocr_sec = float(row[3])
        current_x = abs_ocr_sec - first_ocr_sec
        
        # --- THE SANITY CHECK ---
        if i == 0:
            filtered_x.append(current_x)
            filtered_y.append(vid_sec)
            continue

        # Calculate how much time passed in the Log vs the Video
        delta_log = current_x - last_good_x
        delta_vid = vid_sec - last_good_y
        
        # A 'slope' of 1.0 is perfect. We allow 0.8 to 1.2 to account for drift.
        # If the log jumps 3 hours but video only jumps 17 mins, slope will be ~10.0 (OUTLIER)
        if delta_log > 0:
            slope = delta_log / delta_vid
            if 0.5 < slope < 2.0:  # Very generous window to catch the 18:03 error
                filtered_x.append(current_x)
                filtered_y.append(vid_sec)
                last_good_x = current_x
                last_good_y = vid_sec
            else:
                if debug:
                    print(f"🚩 Outlier Ignored: Video {vid_sec}s said Log was {abs_ocr_sec}s (Slope: {slope:.2f})")
        else:
            if debug:
                print(f"⚠️ Non-increasing point ignored at Video {vid_sec}s")

    return PchipInterpolator(filtered_x, filtered_y)

def extract_synced_clips(log_file, video_file, base_output_folder, drift_data, target_tags):
    if not os.path.exists(base_output_folder): os.makedirs(base_output_folder)
    time_model = build_drift_model(drift_data)
    
    # We will store intervals as (start_dt, end_dt, bee_id)
    clip_intervals = []
    in_sequence = False
    current_phase = "IDLE" 
    current_bee_id = None
    seq_start_time = None
    log_start_time = None
    
    # Convert tags to strings for comparison
    target_tags = [str(t) for t in target_tags]

    with open(log_file, 'r') as file:
        for line in file:
            clean_line = line.strip()
            if not clean_line.startswith('['): continue
                
            try:
                current_time = datetime.strptime(clean_line[1:20], "%Y-%m-%d %H:%M:%S")
            except ValueError: continue

            # --- 1. RECORDING SYNC POINT ---
            if "Recording started:" in clean_line and ".mp4" in clean_line:
                log_start_time = current_time
                continue

            # --- 2. MULTI-TAG DETECTION ---
            if "confirmed. Sending FEED command." in clean_line:
                detected_tag = None
                for t in target_tags:
                    if f"Tag {t} confirmed" in clean_line:
                        detected_tag = t
                        break
                
                if detected_tag and current_phase in ["IDLE", "RETRACTING"]:
                    # Handle Interruption during retraction
                    if current_phase == "RETRACTING":
                        clip_intervals.append((seq_start_time, current_time, current_bee_id))
                    
                    in_sequence = True
                    seq_start_time = current_time
                    current_bee_id = detected_tag
                    current_phase = "EJECTING"
                continue

            # --- 3. PHASE TRANSITIONS (The Anchors) ---
            if in_sequence:
                if "Starting HOLD" in clean_line:
                    current_phase = "HOLDING"

                elif "Hold time over. Retracting." in clean_line:
                    current_phase = "RETRACTING"

                elif "Retract Complete" in clean_line or "Entering WAITING" in clean_line:
                    clip_intervals.append((seq_start_time, current_time, current_bee_id))
                    in_sequence = False
                    current_phase = "IDLE"
                    current_bee_id = None

    if not log_start_time:
        print("Error: Could not find 'Recording started:' line.")
        return

    print(f"Starting Phase 2. Cutting {len(clip_intervals)} clips...")

    # Dictionary to keep track of clip counts per bee for naming: { '667': 1, '183': 5 }
    bee_clip_counts = {t: 0 for t in target_tags}
    valid_clips = 0

    for start_dt, end_dt, bee_id in clip_intervals:
        log_start_sec = (start_dt - log_start_time).total_seconds()
        log_end_sec = (end_dt - log_start_time).total_seconds()
        
        # Apply drift model and 1s padding
        vid_start_sec = max(0, float(time_model(log_start_sec)) - 1.0)
        vid_end_sec = float(time_model(log_end_sec)) + 1.0
        duration = vid_end_sec - vid_start_sec
        
        # Folder management: extracted_clips/bee_667/
        bee_folder = os.path.join(base_output_folder, f"bee_{bee_id}")
        os.makedirs(bee_folder, exist_ok=True)
        
        # Filename: bee_667_clip_0001.mp4
        bee_clip_counts[bee_id] += 1
        clip_name = f"bee_{bee_id}_clip_{bee_clip_counts[bee_id]:04d}.mp4"
        output_path = os.path.join(bee_folder, clip_name)
        
        command = [
            'ffmpeg', '-y', '-ss', f"{vid_start_sec:.3f}", 
            '-i', video_file, '-t', f"{duration:.3f}", 
            '-c', 'copy', output_path
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
            valid_clips += 1
            print(f"Saved: {clip_name} to {bee_folder}")
        except subprocess.CalledProcessError:
            print(f"Error cutting {clip_name}")

    print("-" * 30)
    print(f"Done! {valid_clips} clips organized by Bee ID.")


# Electrode analysis
def analyze_baselines(file_path):
    # Buckets to store raw numbers
    data = {
        'WAITING': {'B': [], 'M': [], 'F': []},
        'HOLDING': {'B': [], 'M': [], 'F': []}
    }

    # Captures B/M/F raw values and state. Supports both old syntax and new syntax with Pattern field.
    pattern = SENSOR_LINE_PATTERN

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                b_val = int(match.group(1))
                m_val = int(match.group(2))
                f_val = int(match.group(3))
                state = match.group(4)

                if state in data:
                    data[state]['B'].append(b_val)
                    data[state]['M'].append(m_val)
                    data[state]['F'].append(f_val)

    print("=" * 50)
    print(f"{'SENSOR BASELINE ANALYSIS':^50}")
    print("=" * 50)

    for state in ['WAITING', 'HOLDING']:
        print(f"\n>>> STATE: {state} <<<")
        for sensor in ['B', 'M', 'F']:
            values = data[state][sensor]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                print(f"  Sensor {sensor}: Avg={mean_val:>7.2f} | Std={std_val:>5.2f} | Range=[{min_val}, {max_val}]")
            else:
                print(f"  Sensor {sensor}: No data found.")
    
    print("\n" + "=" * 50)

def plot_full_cycle_averages(file_path):
    # 1. State Mapping for the X-Axis (Total width 0.0 to 1.0)
    # We allocate specific 'slots' for each phase
    PHASES = {
        'WAIT_PRE':  (0.00, 0.15),
        'EJECTING':  (0.15, 0.40),
        'HOLDING':   (0.40, 0.70),
        'RETRACTING':(0.70, 0.90),
        'WAIT_POST': (0.90, 1.00)
    }

    cycles = []
    current_cycle = {p: [] for p in PHASES}
    active_state = 'WAIT_PRE'
    is_capturing = False

    # Captures B/M/F raw values and state. Supports both old syntax and new syntax with Pattern field.
    pattern = SENSOR_LINE_PATTERN

    # 2. Extract Cycles
    with open(file_path, 'r') as f:
        for line in f:
            # Detect Command/Status Markers to switch internal "Phase"
            if "CMD: Feed Sequence Start" in line:
                is_capturing = True
                active_state = 'EJECTING'
                continue
            elif "STATUS: Full (B+M+F)" in line or "Starting HOLD" in line:
                active_state = 'HOLDING'
                continue
            elif "STATUS: Hold time over" in line:
                active_state = 'RETRACTING'
                continue
            elif "STATUS: Retract Complete" in line:
                active_state = 'WAIT_POST'
                # Give it a few more lines of wait before closing cycle
                continue

            match = pattern.search(line)
            if match and is_capturing:
                b, m, f_val, state = int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
                
                # Check for cycle end (transition from WAIT_POST to next IDLE/WAIT)
                if active_state == 'WAIT_POST' and len(current_cycle['WAIT_POST']) > 20:
                    cycles.append(current_cycle)
                    current_cycle = {p: [] for p in PHASES}
                    is_capturing = False
                    active_state = 'WAIT_PRE'
                
                current_cycle[active_state].append([b, m, f_val])

    # 3. Processing & Interpolation
    x_common = np.linspace(0, 1, 1000)
    sensor_data = [ [] for _ in range(3) ] # B, M, F

    for cyc in cycles:
        for s_idx in range(3):
            full_y = []
            full_x = []
            for phase, (x_start, x_end) in PHASES.items():
                data = np.array(cyc[phase])
                if len(data) == 0: continue
                
                y_vals = data[:, s_idx]
                x_vals = np.linspace(x_start, x_end, len(y_vals))
                full_y.extend(y_vals)
                full_x.extend(x_vals)
            
            # Interpolate the whole stitched cycle onto the 1000-point grid
            sensor_data[s_idx].append(np.interp(x_common, full_x, full_y))

    # 4. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = ['red', 'green', 'blue']
    labels = ['Back Electrode (B)', 'Middle Electrode (M)', 'Front Electrode (F)']

    for i in range(3):
        all_runs = np.array(sensor_data[i])
        avg_run = np.mean(all_runs, axis=0)
        
        # Plot individual cycles faintly
        for run in all_runs:
            axes[i].plot(x_common, run, color=colors[i], alpha=0.1)
        
        # Plot average
        axes[i].plot(x_common, avg_run, color=colors[i], linewidth=2.5, label='Average')
        axes[i].set_ylabel("Raw Reading")
        axes[i].set_title(labels[i])
        axes[i].grid(True, alpha=0.3)

        # Add Vertical Phase Markers
        for phase, (x_start, x_end) in PHASES.items():
            axes[i].axvline(x=x_start, color='black', linestyle='--', alpha=0.5)
            if i == 0: # Label phases on top plot only
                axes[i].text(x_start + 0.01, axes[i].get_ylim()[1]*0.9, phase, fontsize=9)

    axes[2].set_xlabel("Normalized Cycle Time (Sequence: Eject -> Hold -> Retract)")
    plt.tight_layout()
    plt.show()

def dhms_to_seconds(dhms_str):
    """Parses 'D:H:M:S' or 'H:M:S' into total seconds."""
    parts = list(map(int, dhms_str.split(':')))
    if len(parts) == 4:
        d, h, m, s = parts
        return (d * 86400) + (h * 3600) + (m * 60) + s
    elif len(parts) == 3:
        h, m, s = parts
        return (h * 3600) + (m * 60) + s
    return 0

def seconds_to_hms(total_seconds):
    """Converts seconds to standard H:M:S for video players."""
    total_seconds = round(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def map_log_to_video(first_log_ts, event_log_ts, total_exp_dhms, total_vid_hms):
    """
    Maps a Y-M-D H:M:S log timestamp to a video H:M:S timestamp.
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    
    # 1. Calculate Real-World Elapsed Time
    start_dt = datetime.strptime(first_log_ts, fmt)
    event_dt = datetime.strptime(event_log_ts, fmt)
    real_elapsed_sec = (event_dt - start_dt).total_seconds()
    
    # 2. Convert Durations to Seconds
    exp_total_sec = dhms_to_seconds(total_exp_dhms)
    vid_total_sec = dhms_to_seconds(total_vid_hms)
    
    # 3. Calculate Scale (Drift) Ratio
    if exp_total_sec == 0: return "0:00:00"
    ratio = vid_total_sec / exp_total_sec
    
    # 4. Map the time
    video_sec = real_elapsed_sec * ratio
    
    # Return 0:00:00 if the event happened before the start (sanity check)
    if video_sec < 0: return "0:00:00"
    
    return seconds_to_hms(video_sec)


def plot_complete_bee_analysis(file_path, target_tags, duration_str=None):
    # 1. Setup & Duration Parsing
    time_limit = None
    if duration_str:
        h, m, s = map(int, duration_str.split(':'))
        time_limit = timedelta(hours=h, minutes=m, seconds=s)

    # Data Containers
    times = []
    f_raw = []
    f_bin_normal = []
    f_bin_autofeed = []
    soft_stop_binary = [] 
    steps_values = [] 
    bee_triggers = {str(t): [] for t in target_tags}
    secondary_events = [] 
    
    current_steps_val = 0
    in_auto_feed = False
    start_time = None
    
    # Regex Patterns
    time_pat = TIME_PATTERN
    f_pat = F_VALUE_PATTERN
    sec_pat = re.compile(r'Possible feeding detected at secondary zone:\s+([\d.]+)')
    steps_pat = re.compile(r'number_of_steps set:\s+(\d+)')
    
    # 2. Single-Pass Data Extraction
    with open(file_path, 'r') as f:
        for line in f:
            t_match = time_pat.search(line)
            if not t_match: continue
            
            curr_time = datetime.strptime(t_match.group(1), "%Y-%m-%d %H:%M:%S")
            if start_time is None: start_time = curr_time
            if time_limit and (curr_time - start_time) > time_limit: break

            if "AUTO FEED: Triggered" in line or "| AUTO_FEED |" in line:
                in_auto_feed = True
            elif "Retract Complete" in line or "Entering WAITING" in line:
                in_auto_feed = False

            s_match = steps_pat.search(line)
            if s_match:
                current_steps_val = int(s_match.group(1))

            if "SENSORS:" in line:
                f_match = f_pat.search(line)
                if f_match:
                    val = int(f_match.group(1))
                    times.append(curr_time)
                    f_raw.append(val)
                    
                    is_on = 1 if val > F_THRESHOLD else 0
                    if in_auto_feed and is_on:
                        f_bin_autofeed.append(1); f_bin_normal.append(0)
                    elif is_on:
                        f_bin_autofeed.append(0); f_bin_normal.append(1)
                    else:
                        f_bin_autofeed.append(0); f_bin_normal.append(0)
                    
                    steps_values.append(current_steps_val)
                    for tag in bee_triggers: bee_triggers[tag].append(0)
                    soft_stop_binary.append(0)

            sec_match = sec_pat.search(line)
            if sec_match: secondary_events.append((curr_time, float(sec_match.group(1))))

            if "confirmed. Sending FEED command." in line:
                for tag in bee_triggers:
                    if f"Tag {tag} confirmed" in line:
                        if bee_triggers[tag]: bee_triggers[tag][-1] = 1

            if "STATUS: Soft Stop (Distance). Starting HOLD." in line:
                if soft_stop_binary: soft_stop_binary[-1] = 1

    # 3. Create the Subplots
    # We use specs to allow a secondary Y-axis on the 5th panel
    fig = make_subplots(
        rows=5, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        subplot_titles=("Bee Triggers", "F-Sensor Binary (Normal vs Auto)", "F-Sensor Raw", "Secondary Area Activity", "Soft Stop Trigger"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # Panel 1: Bee Triggers
    for tag, vals in bee_triggers.items():
        fig.add_trace(go.Scatter(x=times, y=vals, name=f"Bee {tag}", mode='lines'), row=1, col=1)

    # Panel 2: F-Sensor Binary (Color Coded)
    fig.add_trace(go.Scatter(x=times, y=f_bin_normal, name="Normal Bump", line_shape='hv', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=f_bin_autofeed, name="Auto-Feed Drinking", line_shape='hv', line=dict(color='orange'), fill='tozeroy'), row=2, col=1)

    # Panel 3: F-Sensor Raw
    fig.add_trace(go.Scatter(x=times, y=f_raw, name="Raw F-Value", line=dict(color='teal', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=times, y=[F_THRESHOLD]*len(times), name=f"Threshold ({F_THRESHOLD})", line=dict(color='red', dash='dash')), row=3, col=1)

    # Panel 4: Secondary Zone (Chains)
    # Mapping chains to 1s
    chain_binary = [0] * len(times)
    last_t = None
    for t, val in secondary_events:
        if last_t is None or (t - last_t).total_seconds() > 1.0:
             for idx, mt in enumerate(times):
                if mt >= t:
                    chain_binary[idx] = 1
                    break
        last_t = t
    fig.add_trace(go.Scatter(x=times, y=chain_binary, name="Secondary Chain", line_shape='vh', line=dict(color='darkorange')), row=4, col=1)

    # Panel 5: Soft Stop (Left Y) and Steps (Right Y)
    fig.add_trace(go.Scatter(x=times, y=soft_stop_binary, name="Soft Stop", line_shape='vh', line=dict(color='crimson')), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=times, y=steps_values, name="Steps Set", line_shape='hv', line=dict(color='blue', dash='dot')), row=5, col=1, secondary_y=True)

    # 4. Interactive Enhancements
    fig.update_layout(
        height=1200, 
        title_text=f"Interactive Bee Analysis: {file_path}",
        showlegend=True,
        hovermode="x unified" # Shows all panel data at a single time-slice when hovering
    )

    # Add a Range Slider at the bottom (Crucial for 89-hour files)
    fig.update_xaxes(rangeslider_visible=True, row=5, col=1)

    # 5. THE MAGIC SAVE LINE
    output_name = file_path.replace(".txt", "_interactive.html")
    fig.write_html(output_name)
    print(f"Interactive graph saved to: {output_name}")
    
    # Also show it immediately
    fig.show()

def extract_feed_sequence_clips_no_drift_from_csv(
    sequence_csv,
    log_file,
    video_file,
    output_folder="feed_sequence_clips",
    pre_pad_sec=2.0,
    post_pad_sec=3.0,
    accurate_cut=False,
    skip_zero_duration=True,
    recording_start_override=None
):
    """
    Extracts feed sequence clips using feed_sequence_durations.csv.

    Assumption:
        No drift between log and video.

    Video time is calculated as:
        video_time = CSV Tag Time - log Recording started time

    CSV must contain:
        Tag ID,
        Tag Time,
        Eject Time (s),
        Hold Time (s),
        Retract Time (s),
        Detected During Retract

    If the log does not contain a 'Recording started:' line, you can manually pass:
        recording_start_override="2026-05-04 12:00:00"
    """

    os.makedirs(output_folder, exist_ok=True)

    # -------------------------------------------------
    # 1. Find video/log zero time from TXT log
    # -------------------------------------------------
    recording_start_time = None

    if recording_start_override is not None:
        recording_start_time = datetime.strptime(
            recording_start_override,
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"Using manual recording start time: {recording_start_time}")

    else:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Recording started:" in line and ".mp4" in line:
                    m = TIME_PATTERN.search(line)
                    if m:
                        recording_start_time = datetime.strptime(
                            m.group(1),
                            "%Y-%m-%d %H:%M:%S"
                        )
                        break

        if recording_start_time is None:
            print("ERROR: Could not find 'Recording started:' line in log.")
            print("Use recording_start_override='YYYY-MM-DD HH:MM:SS' if needed.")
            return

        print(f"Recording start time: {recording_start_time}")

    # -------------------------------------------------
    # 2. Read CSV and cut clips
    # -------------------------------------------------
    summary_path = os.path.join(output_folder, "clip_extraction_summary.csv")

    total_rows = 0
    saved_clips = 0
    skipped_rows = 0

    with open(sequence_csv, "r", newline="", encoding="utf-8", errors="ignore") as infile, \
         open(summary_path, "w", newline="", encoding="utf-8") as summary_file:

        reader = csv.DictReader(infile)
        writer = csv.writer(summary_file)

        writer.writerow([
            "Clip Index",
            "Tag ID",
            "Tag Time",
            "Raw Video Start (s)",
            "Video Start (s)",
            "Clip Duration (s)",
            "Eject Time (s)",
            "Hold Time (s)",
            "Retract Time (s)",
            "Detected During Retract",
            "Output File",
            "Status"
        ])

        for row in reader:
            total_rows += 1

            try:
                tag_id = row["Tag ID"]
                tag_time_str = row["Tag Time"]

                tag_time = datetime.strptime(
                    tag_time_str,
                    "%Y-%m-%d %H:%M:%S"
                )

                eject_time = float(row["Eject Time (s)"])
                hold_time = float(row["Hold Time (s)"])
                retract_time = float(row["Retract Time (s)"])
                detected_during_retract = row.get("Detected During Retract", "")

            except Exception as e:
                skipped_rows += 1
                writer.writerow([
                    total_rows, "", "", "", "", "", "", "", "", "",
                    "", f"Skipped: CSV parse error: {e}"
                ])
                continue

            sequence_duration = eject_time + hold_time + retract_time

            if skip_zero_duration and sequence_duration <= 0:
                skipped_rows += 1
                writer.writerow([
                    total_rows,
                    tag_id,
                    tag_time_str,
                    "",
                    "",
                    "",
                    eject_time,
                    hold_time,
                    retract_time,
                    detected_during_retract,
                    "",
                    "Skipped: zero duration"
                ])
                continue

            raw_video_start = (tag_time - recording_start_time).total_seconds()

            # If the event is very close to video start, only use available pre-padding.
            actual_pre_pad = min(pre_pad_sec, max(0.0, raw_video_start))
            video_start = max(0.0, raw_video_start - pre_pad_sec)
            clip_duration = actual_pre_pad + sequence_duration + post_pad_sec

            if clip_duration <= 0:
                skipped_rows += 1
                writer.writerow([
                    total_rows,
                    tag_id,
                    tag_time_str,
                    round(raw_video_start, 3),
                    round(video_start, 3),
                    round(clip_duration, 3),
                    eject_time,
                    hold_time,
                    retract_time,
                    detected_during_retract,
                    "",
                    "Skipped: invalid clip duration"
                ])
                continue

            bee_folder = os.path.join(output_folder, f"bee_{tag_id}")
            os.makedirs(bee_folder, exist_ok=True)

            clip_name = f"bee_{tag_id}_feed_{total_rows:04d}.mp4"
            output_path = os.path.join(bee_folder, clip_name)

            if accurate_cut:
                # More accurate timing, slower, re-encodes video.
                command = [
                    "ffmpeg", "-y",
                    "-ss", f"{video_start:.3f}",
                    "-i", video_file,
                    "-t", f"{clip_duration:.3f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-c:a", "aac",
                    output_path
                ]
            else:
                # Fast, no re-encoding, but may cut near keyframes.
                command = [
                    "ffmpeg", "-y",
                    "-ss", f"{video_start:.3f}",
                    "-i", video_file,
                    "-t", f"{clip_duration:.3f}",
                    "-c", "copy",
                    output_path
                ]

            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode == 0 and os.path.exists(output_path):
                    saved_clips += 1
                    status = "Saved"
                    print(
                        f"Saved {clip_name} | "
                        f"start={video_start:.2f}s | "
                        f"duration={clip_duration:.2f}s"
                    )
                else:
                    skipped_rows += 1
                    status = "ffmpeg failed"
                    print(f"ERROR cutting {clip_name}")
                    print(result.stderr[-500:])

                writer.writerow([
                    total_rows,
                    tag_id,
                    tag_time_str,
                    round(raw_video_start, 3),
                    round(video_start, 3),
                    round(clip_duration, 3),
                    eject_time,
                    hold_time,
                    retract_time,
                    detected_during_retract,
                    output_path,
                    status
                ])

            except Exception as e:
                skipped_rows += 1
                writer.writerow([
                    total_rows,
                    tag_id,
                    tag_time_str,
                    round(raw_video_start, 3),
                    round(video_start, 3),
                    round(clip_duration, 3),
                    eject_time,
                    hold_time,
                    retract_time,
                    detected_during_retract,
                    output_path,
                    f"Exception: {e}"
                ])

    print("-" * 40)
    print(f"Rows in CSV: {total_rows}")
    print(f"Saved clips: {saved_clips}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Summary CSV: {summary_path}")

def extract_possible_feeding_clips_from_csv(
    possible_feeding_csv,
    video_file,
    output_folder="possible_feeding_clips",
    recording_start_time=None,
    log_file=None,
    min_duration_sec=4.0,
    pre_pad_sec=1.0,
    post_pad_sec=1.0,
    accurate_cut=False
):
    """
    Extracts video clips for possible-feeding events from a CSV file.

    CSV format expected:
        Start Timestamp,
        Min Intensity,
        Max Intensity,
        Avg Intensity,
        Log Count,
        Duration (s)

    Selection rule:
        Duration (s) > min_duration_sec

    No drift assumption:
        video_time = event_timestamp - recording_start_time

    You must provide either:
        1. recording_start_time="YYYY-MM-DD HH:MM:SS"
    or:
        2. log_file="your_log.txt", where the log contains a 'Recording started:' line.
    """

    os.makedirs(output_folder, exist_ok=True)

    # -------------------------------------------------
    # 1. Find recording/video zero time
    # -------------------------------------------------
    if recording_start_time is not None:
        video_zero_time = datetime.strptime(
            recording_start_time,
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"Using manual recording start time: {video_zero_time}")

    elif log_file is not None:
        video_zero_time = None

        time_pattern = re.compile(r'^\[([\d\-\s:]+)\]')

        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Recording started:" in line and ".mp4" in line:
                    m = time_pattern.search(line)
                    if m:
                        video_zero_time = datetime.strptime(
                            m.group(1),
                            "%Y-%m-%d %H:%M:%S"
                        )
                        break

        if video_zero_time is None:
            print("ERROR: Could not find 'Recording started:' line in log.")
            return

        print(f"Recording start time found from log: {video_zero_time}")

    else:
        print("ERROR: Provide either recording_start_time or log_file.")
        return

    # -------------------------------------------------
    # 2. Prepare summary CSV
    # -------------------------------------------------
    summary_path = os.path.join(output_folder, "possible_feeding_clip_summary.csv")

    total_rows = 0
    matched_rows = 0
    saved_clips = 0
    skipped_rows = 0

    with open(possible_feeding_csv, "r", newline="", encoding="utf-8", errors="ignore") as infile, \
         open(summary_path, "w", newline="", encoding="utf-8") as summary_file:

        reader = csv.DictReader(infile)
        writer = csv.writer(summary_file)

        writer.writerow([
            "CSV Row",
            "Start Timestamp",
            "Duration (s)",
            "Avg Intensity",
            "Video Start (s)",
            "Clip Duration (s)",
            "Output File",
            "Status"
        ])

        # -------------------------------------------------
        # 3. Process CSV rows
        # -------------------------------------------------
        for row in reader:
            total_rows += 1

            try:
                start_time_str = row["Start Timestamp"]
                event_start_time = datetime.strptime(
                    start_time_str,
                    "%Y-%m-%d %H:%M:%S"
                )

                duration_sec = float(row["Duration (s)"])
                avg_intensity = row.get("Avg Intensity", "")

            except Exception as e:
                skipped_rows += 1
                writer.writerow([
                    total_rows,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"Skipped: CSV parse error: {e}"
                ])
                continue

            # Important: strictly greater than 4 seconds
            if duration_sec <= min_duration_sec:
                continue

            matched_rows += 1

            raw_video_start = (event_start_time - video_zero_time).total_seconds()

            # Handle events very close to the beginning of the video
            actual_pre_pad = min(pre_pad_sec, max(0.0, raw_video_start))
            video_start = max(0.0, raw_video_start - pre_pad_sec)

            clip_duration = actual_pre_pad + duration_sec + post_pad_sec

            if clip_duration <= 0:
                skipped_rows += 1
                writer.writerow([
                    total_rows,
                    start_time_str,
                    duration_sec,
                    avg_intensity,
                    round(video_start, 3),
                    round(clip_duration, 3),
                    "",
                    "Skipped: invalid clip duration"
                ])
                continue

            clip_name = (
                f"possible_feed_{matched_rows:04d}_"
                f"{event_start_time.strftime('%H%M%S')}_"
                f"{duration_sec:.1f}s.mp4"
            )

            output_path = os.path.join(output_folder, clip_name)

            # -------------------------------------------------
            # 4. Cut clip with ffmpeg
            # -------------------------------------------------
            if accurate_cut:
                # More accurate, slower, re-encodes
                command = [
                    "ffmpeg", "-y",
                    "-ss", f"{video_start:.3f}",
                    "-i", video_file,
                    "-t", f"{clip_duration:.3f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-c:a", "aac",
                    output_path
                ]
            else:
                # Faster, no re-encoding, may cut near keyframes
                command = [
                    "ffmpeg", "-y",
                    "-ss", f"{video_start:.3f}",
                    "-i", video_file,
                    "-t", f"{clip_duration:.3f}",
                    "-c", "copy",
                    output_path
                ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0 and os.path.exists(output_path):
                saved_clips += 1
                status = "Saved"

                print(
                    f"Saved {clip_name} | "
                    f"start={video_start:.2f}s | "
                    f"duration={clip_duration:.2f}s"
                )

            else:
                skipped_rows += 1
                status = "ffmpeg failed"
                print(f"ERROR cutting {clip_name}")
                print(result.stderr[-500:])

            writer.writerow([
                total_rows,
                start_time_str,
                duration_sec,
                avg_intensity,
                round(video_start, 3),
                round(clip_duration, 3),
                output_path,
                status
            ])

    print("-" * 40)
    print(f"Total CSV rows: {total_rows}")
    print(f"Rows with Duration > {min_duration_sec}: {matched_rows}")
    print(f"Saved clips: {saved_clips}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Summary CSV: {summary_path}")

def assign_label_levels(event_positions, min_separation_sec=120):
    """
    Assigns each event to a vertical label level so nearby labels do not overlap.

    Parameters
    ----------
    event_positions : list of float
        Event x-positions in seconds.

    min_separation_sec : float
        Minimum horizontal spacing required between labels on the same row.

    Returns
    -------
    levels : list of int
        Row index for each event label.
    """
    levels = []
    level_last_x = []

    for x in event_positions:
        placed = False

        for level_idx in range(len(level_last_x)):
            if x - level_last_x[level_idx] >= min_separation_sec:
                levels.append(level_idx)
                level_last_x[level_idx] = x
                placed = True
                break

        if not placed:
            levels.append(len(level_last_x))
            level_last_x.append(x)

    return levels

def plot_verified_feeding_times_from_folder_by_duration(
    folder_path,
    experiment_start_time_str,
    experiment_duration_str,
    output_plot_path=None,
    extensions=(".mp4", ".avi", ".mov", ".mkv"),
    title="Verified Feeding Events",
    show_event_time_labels=True,
):
    """
    Scans a folder of manually verified feeding clips, extracts feeding times
    from filenames, converts them to elapsed experiment time, and plots them.

    Expected filename style:
        possible_feed_0001_121640_8.0s.mp4

    The 6-digit block is interpreted as HHMMSS:
        121640 -> 12:16:40

    X-axis:
        Main x-axis shows elapsed experiment time.
        Optional event labels show actual feeding clock times.
    """

    if not os.path.isdir(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        return []

    # -------------------------------------------------
    # 1. Parse experiment start and duration
    # -------------------------------------------------
    try:
        experiment_start_dt = datetime.strptime(
            experiment_start_time_str,
            "%H:%M:%S"
        )
    except ValueError:
        print("ERROR: experiment_start_time_str must be in HH:MM:SS format.")
        return []

    try:
        h, m, s = map(int, experiment_duration_str.split(":"))
        experiment_duration_sec = h * 3600 + m * 60 + s
    except ValueError:
        print("ERROR: experiment_duration_str must be in HH:MM:SS format.")
        return []

    if experiment_duration_sec <= 0:
        print("ERROR: Experiment duration must be greater than zero.")
        return []

    # -------------------------------------------------
    # 2. Collect video files
    # -------------------------------------------------
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith(extensions)
    ]

    if not files:
        print("No video files found in folder.")
        return []

    # -------------------------------------------------
    # 3. Extract HHMMSS time from filenames
    # -------------------------------------------------
    time_pattern = re.compile(r'(?<!\d)(\d{6})(?!\d)')

    extracted_events = []
    unmatched_files = []

    for fname in sorted(files):
        matches = time_pattern.findall(fname)

        valid_times = []

        for hhmmss in matches:
            try:
                clock_dt = datetime.strptime(hhmmss, "%H%M%S")
                valid_times.append(clock_dt)
            except ValueError:
                pass

        if not valid_times:
            unmatched_files.append(fname)
            continue

        # Use the last valid HHMMSS block.
        # This is safer for names like:
        # possible_feed_0001_121640_8.0s.mp4
        event_clock_dt = valid_times[-1]

        # Handle midnight rollover.
        if event_clock_dt < experiment_start_dt:
            event_clock_dt = event_clock_dt + timedelta(days=1)

        elapsed_sec = (event_clock_dt - experiment_start_dt).total_seconds()

        extracted_events.append((fname, event_clock_dt, elapsed_sec))

    if unmatched_files:
        print("Warning: Could not extract valid HHMMSS time from these files:")
        for f in unmatched_files:
            print("  ", f)

    if not extracted_events:
        print("No valid feeding times extracted from filenames.")
        return []

    # -------------------------------------------------
    # 4. Filter events within experiment duration
    # -------------------------------------------------
    filtered_events = [
        event for event in extracted_events
        if 0 <= event[2] <= experiment_duration_sec
    ]

    filtered_events.sort(key=lambda x: x[2])

    if not filtered_events:
        print("No extracted feeding events fall within the experiment duration.")
        return []

    # -------------------------------------------------
    # 5. Prepare plot
    # -------------------------------------------------
    event_elapsed_seconds = [event[2] for event in filtered_events]
    event_clock_labels = [event[1].strftime("%H:%M:%S") for event in filtered_events]

    y_vals = [1] * len(event_elapsed_seconds)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.scatter(event_elapsed_seconds, y_vals, s=60)

    for elapsed in event_elapsed_seconds:
        ax.axvline(elapsed, ymin=0.15, ymax=0.85, linestyle="--", alpha=0.35)

    ax.set_xlim(0, experiment_duration_sec)
    ax.set_ylim(0.75, 1.25)
    ax.set_yticks([1])
    ax.set_yticklabels(["Feeding"])

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Elapsed experiment time")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # -------------------------------------------------
    # 6. Format main x-axis as elapsed HH:MM:SS
    # -------------------------------------------------
    def seconds_to_hms_for_axis(x, pos):
        x = int(round(x))
        hours = x // 3600
        minutes = (x % 3600) // 60
        seconds = x % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(seconds_to_hms_for_axis))

    # -------------------------------------------------
    # 7. Add feeding clock times on the x-axis
    # -------------------------------------------------
    if show_event_time_labels:
            # You can tune this:
            # if labels still collide, increase this number
            min_label_spacing_sec = 120

            label_levels = assign_label_levels(
                event_elapsed_seconds,
                min_separation_sec=min_label_spacing_sec
            )

            # Base y position for labels, below the main axis
            base_y = 0.72
            level_step = 0.08

            # Extend lower margin depending on number of label rows
            max_level = max(label_levels) if label_levels else 0
            extra_bottom = 0.18 + max_level * 0.05
            plt.subplots_adjust(bottom=min(0.45, extra_bottom))

            for elapsed, clock_label, level in zip(
                event_elapsed_seconds,
                event_clock_labels,
                label_levels
            ):
                label_y = base_y - level * level_step

                # vertical guide line from point to label zone
                ax.plot(
                    [elapsed, elapsed],
                    [1.0, label_y + 0.015],
                    linestyle=":",
                    alpha=0.4
                )

                # text label
                ax.text(
                    elapsed,
                    label_y,
                    clock_label,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=8
                )

    plt.tight_layout()

    if output_plot_path is not None:
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_plot_path}")

    plt.show()

    # -------------------------------------------------
    # 8. Print summary
    # -------------------------------------------------
    print("-" * 40)
    print(f"Video files found: {len(files)}")
    print(f"Successfully parsed feeding times: {len(extracted_events)}")
    print(f"Events within duration: {len(filtered_events)}")
    print(f"Experiment start clock time: {experiment_start_time_str}")
    print(f"Experiment duration: {experiment_duration_str}")
    print("-" * 40)

    for fname, clock_dt, elapsed_sec in filtered_events:
        elapsed_str = seconds_to_hms_for_axis(elapsed_sec, None)
        clock_str = clock_dt.strftime("%H:%M:%S")
        print(f"{elapsed_str} | clock {clock_str} -> {fname}")

    return filtered_events

input_log = 'feed_output_20260504_115703_1920x1080.txt'  # frame rate test log change!!
vid_path = 'feed_output_20260504_115703_1920x1080.mp4'
output_csv = 'feed_sequence_durations.csv'
possible_feeding_csv = 'possible_feeding.csv'

plot_verified_feeding_times_from_folder_by_duration(
    folder_path="possible_feeding_clips",
    experiment_start_time_str="11:57:03",
    experiment_duration_str="07:45:00",
    output_plot_path="verified_feeding_timeline.png",
    show_event_time_labels=True
)