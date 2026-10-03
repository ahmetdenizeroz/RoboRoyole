import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import csv
import re
import os
import subprocess
import cv2
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PySide6.QtCore import QObject, Signal, QThread
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety

# --- CONFIGURATION ---

F_THRESHOLD = 300
TIME_PATTERN = re.compile(r'^\[([\d\-\s:]+)\]')
F_VALUE_PATTERN = re.compile(r'F=(?:ON|OFF)\s+(\d+)')
SENSOR_LINE_PATTERN = re.compile(
    r"(?:\[Hardware\]:\s*)?SENSORS:\s*"
    r"B=(?:ON|OFF)\s+(\d+)\s+"
    r"M=(?:ON|OFF)\s+(\d+)\s+"
    r"F=(?:ON|OFF)\s+(\d+)"
    r"(?:\s+\|\s+Pattern:\s+[^|]+)?"
    r"\s+\|\s+State:\s+(\w+)"
)
STATE_PATTERN = re.compile(r'\|\s*State:\s*(\w+)')

class AnalyzerWorker(QThread):
    finished = Signal()
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

class AnalyzerCore(QObject):
    log_message = Signal(str)

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_message.emit(str(message))

    def parse_time_to_seconds(self, time_str):
        if not time_str.strip(): return None
        parts = time_str.strip().split(':')
        try:
            if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
            elif len(parts) == 2: return int(parts[0])*60 + float(parts[1])
            elif len(parts) == 1: return float(parts[0])
        except ValueError:
            pass
        return None

    def trim_data(self, log_path, vid_path, start_time_str, end_time_str):
        T_start = self.parse_time_to_seconds(start_time_str) or 0.0
        T_end = self.parse_time_to_seconds(end_time_str)

        if not os.path.exists(log_path):
            self.log(f"Error: Log file {log_path} not found.")
            return

        if not os.path.exists(vid_path):
            self.log(f"Error: Video file {vid_path} not found.")
            return

        out_vid = os.path.join(os.path.dirname(vid_path), "trimmed.mp4")
        out_log = os.path.join(os.path.dirname(log_path), "trimmed.txt")

        self.log(f"Trimming video from {T_start}s to {T_end if T_end else 'end'}...")
        cmd = ["ffmpeg", "-y", "-ss", str(T_start), "-i", vid_path]
        if T_end is not None:
            cmd.extend(["-t", str(T_end - T_start)])
        cmd.extend(["-c", "copy", out_vid])

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.log(f"Video trimmed and saved to {out_vid}")
        except Exception as e:
            self.log(f"Error trimming video: {e}")
            return

        self.log("Trimming log...")
        recording_start_time = None
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = TIME_PATTERN.search(line)
                if m:
                    recording_start_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    break

        if not recording_start_time:
            self.log("Error: Could not determine start time of log.")
            return

        abs_start = recording_start_time + timedelta(seconds=T_start)
        abs_end = recording_start_time + timedelta(seconds=T_end) if T_end is not None else None

        trimmed_lines = 0
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(out_log, 'w', encoding='utf-8') as outfile:
            for line in infile:
                m = TIME_PATTERN.search(line)
                if m:
                    current_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    if current_time < abs_start:
                        continue
                    if abs_end and current_time > abs_end:
                        continue
                outfile.write(line)
                trimmed_lines += 1

        self.log(f"Log trimmed ({trimmed_lines} lines kept) and saved to {out_log}")

    def validate_aruco_tags(self, file_path, expected_tags):
        expected_set = set(str(t) for t in expected_tags)
        detected_set = set()
        tag_pattern = re.compile(r"Tag (\d+) confirmed")

        if not os.path.exists(file_path):
            self.log(f"Error: Log file {file_path} not found.")
            return

        with open(file_path, 'r') as file:
            for line in file:
                if "confirmed. Sending FEED command." in line:
                    match = tag_pattern.search(line)
                    if match:
                        detected_set.add(match.group(1))

        ghost_tags = detected_set - expected_set
        missing_tags = expected_set - detected_set

        self.log("\n" + "="*40)
        self.log("      ARUCO TAG VALIDATION REPORT")
        self.log("="*40)
        self.log(f"Total Expected: {len(expected_set)}")
        self.log(f"Total Detected: {len(detected_set)}")
        self.log("-" * 40)

        if not ghost_tags and not missing_tags:
            self.log("SUCCESS: All expected tags were detected and no ghost tags found.")
        else:
            if ghost_tags:
                self.log(f"GHOST TAGS (Detected but not expected):")
                for tag in sorted(ghost_tags, key=int):
                    self.log(f"   - ID: {tag}")
            else:
                self.log("No Ghost Tags detected.")

            self.log("")

            if missing_tags:
                self.log(f"MISSING TAGS (Expected but never detected):")
                for tag in sorted(missing_tags, key=int):
                    self.log(f"   - ID: {tag}")
            else:
                self.log("All expected bees were detected at least once.")
        
        self.log("="*40 + "\n")
        return list(detected_set)

    def calculate_total_log_time(self, file_path):
        start_time = None
        end_time = None

        if not os.path.exists(file_path):
            self.log(f"Error: Log file {file_path} not found.")
            return

        with open(file_path, 'r') as file:
            for line in file:
                clean_line = line.strip()
                if not clean_line.startswith('['):
                    continue
                time_str = clean_line[1:20]
                try:
                    current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                    if start_time is None:
                        start_time = current_time
                    end_time = current_time
                except ValueError:
                    continue

        if start_time and end_time:
            time_difference = end_time - start_time
            total_seconds = int(time_difference.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            self.log("--- Log Duration Summary ---")
            self.log(f"Log Started:  {start_time}")
            self.log(f"Log Ended:    {end_time}")
            self.log("-" * 26)
            self.log(f"Total Uptime: {hours} hours, {minutes} minutes, {seconds} seconds")
        else:
            self.log("Could not find valid timestamps in the log file.")

    def number_of_lines(self, file_path, target_tags):
        total_lines = 0
        tag_counts = {str(tag): 0 for tag in target_tags}
        state_counts = {'IDLE': 0, 'HOLDING': 0, 'WAITING': 0, 'EJECTING': 0, 'RETRACTING': 0}
        auto_feed_count = 0
        motion_feed_count = 0
        
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        with open(file_path, 'r') as file:
            for line in file:
                total_lines += 1
                clean_line = line.strip()
                
                if "Tag None confirmed. Sending FEED command." in clean_line:
                    motion_feed_count += 1
                elif "confirmed. Sending FEED command." in clean_line:
                    for tag in tag_counts.keys():
                        if f"Tag {tag} confirmed" in clean_line:
                            tag_counts[tag] += 1
                            break
                            
                if "AUTO FEED: Triggered" in clean_line:
                    auto_feed_count += 1
                    
                state_match = STATE_PATTERN.search(clean_line)
                if state_match:
                    current_state = state_match.group(1)
                    if current_state in state_counts:
                        state_counts[current_state] += 1

        self.log(f"\n--- Multi-Bee Log Analysis ---")
        self.log(f"Total Lines Processed: {total_lines}")
        self.log("-" * 30)
        self.log("Feed Commands per Bee:")
        for tag, count in tag_counts.items():
            self.log(f"  Bee {tag}: {count} feeds")
        self.log(f"  Auto Feeds: {auto_feed_count}")
        self.log(f"  Motion Detection Feeds: {motion_feed_count}")
        
        self.log("-" * 30)
        self.log("System State Totals:")
        for state, count in state_counts.items():
            self.log(f"  {state}: {count} lines")
        self.log("-" * 30 + "\n")

    def feeding_trigger_vs_time(self, file_path, target_tags):
        tag_data = {str(tag): [] for tag in target_tags}
        times = []
        
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        with open(file_path, 'r') as file:
            for line in file:
                clean_line = line.strip()
                if not clean_line.startswith('['): continue
                try:
                    time_str = clean_line[1:20]
                    current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                    times.append(current_time)
                    
                    for tag in tag_data.keys():
                        phrase = f"Tag {tag} confirmed. Sending FEED command."
                        if phrase in clean_line:
                            tag_data[tag].append(1)
                        else:
                            tag_data[tag].append(0)
                            
                except ValueError: continue

        if not times:
            self.log("No valid timestamps found to plot.")
            return

        num_tags = len(target_tags)
        fig, axes = plt.subplots(num_tags, 1, figsize=(14, 3 * num_tags), sharex=True)
        if num_tags == 1: axes = [axes]

        self.log(f"--- Multi-Tag Activity Summary ---")
        for i, tag in enumerate(tag_data.keys()):
            ax = axes[i]
            values = tag_data[tag]
            total_feeds = sum(values)
            self.log(f"Tag {tag}: {total_feeds} feed commands.")

            ax.plot(times, values, marker='.', linestyle='-', color=plt.cm.tab10(i % 10), markersize=2, alpha=0.7, label=f'Bee {tag}')
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Other', 'FEED'])
            ax.set_ylabel(f'Bee {tag}')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        axes[0].set_title('Feeding Activity Comparison Across Multiple Tags', pad=20, fontweight='bold')
        plt.xlabel('Time')
        plt.tight_layout()
        out_file = 'feeding_trigger_vs_time.png'
        plt.savefig(out_file)
        plt.close()
        self.log(f"Plot saved to {out_file}")

    def time_data_for_feeding(self, input_file, output_file, target_tags):
        target_tags = [str(t) for t in target_tags]

        if not os.path.exists(input_file):
            self.log(f"Error: {input_file} not found.")
            return

        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Tag ID', 'Tag Time', 'Eject Time (s)', 'Hold Time (s)', 'Retract Time (s)', 'Detected During Retract'])
            
            in_sequence = False
            current_phase = "IDLE" 
            current_tag_id = ""
            tag_time_str = ""
            is_retract_tag = False 
            
            eject_start = eject_end = hold_start = hold_end = retract_start = retract_end = None
            sequence_count = 0

            for line in infile:
                clean_line = line.strip()
                if not clean_line.startswith('['): continue
                time_str = clean_line[1:20]
                try: current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError: continue

                is_motion_feed = "Tag None confirmed. Sending FEED command." in clean_line
                is_tag_feed = "confirmed. Sending FEED command." in clean_line and not is_motion_feed
                is_auto_feed = "AUTO FEED: Triggered" in clean_line

                if is_tag_feed or is_auto_feed or is_motion_feed:
                    detected_tag = None
                    if is_motion_feed:
                        detected_tag = "motion detection feeding"
                    elif is_tag_feed:
                        for t in target_tags:
                            if f"Tag {t} confirmed" in clean_line:
                                detected_tag = t
                                break
                    elif is_auto_feed:
                        detected_tag = "autofeed"
                    
                    if detected_tag and current_phase in ["IDLE", "RETRACTING"]:
                        if current_phase == "RETRACTING":
                            self.write_sequence_to_csv(writer, current_tag_id, tag_time_str, eject_start, eject_end, hold_start, hold_end, retract_start, current_time, is_retract_tag)
                            sequence_count += 1
                            is_retract_tag = True
                        else:
                            is_retract_tag = False
                        
                        in_sequence = True
                        current_tag_id = detected_tag
                        tag_time_str = time_str
                        eject_start = current_time
                        eject_end = hold_start = hold_end = retract_start = retract_end = None
                        current_phase = "EJECTING"
                    continue

                if "Starting HOLD" in clean_line and in_sequence:
                    eject_end = hold_start = current_time
                    current_phase = "HOLDING"
                elif "Hold time over. Retracting." in clean_line and in_sequence:
                    hold_end = retract_start = current_time
                    current_phase = "RETRACTING"
                elif "Retract Complete" in clean_line and in_sequence:
                    retract_end = current_time
                    self.write_sequence_to_csv(writer, current_tag_id, tag_time_str, eject_start, eject_end, hold_start, hold_end, retract_start, retract_end, is_retract_tag)
                    sequence_count += 1
                    in_sequence = False
                    current_phase = "IDLE"
                    current_tag_id = tag_time_str = ""

        self.log(f"Extraction complete! {sequence_count} sequences saved to {output_file}.")

    def write_sequence_to_csv(self, writer, tag_id, tag_time, e_start, e_end, h_start, h_end, r_start, r_end, is_retract):
        e_time = (e_end - e_start).total_seconds() if (e_start and e_end) else 0
        h_time = (h_end - h_start).total_seconds() if (h_start and h_end) else 0
        r_time = (r_end - r_start).total_seconds() if (r_start and r_end) else 0
        retract_str = "Yes" if is_retract else "No"
        writer.writerow([tag_id, tag_time, e_time, h_time, r_time, retract_str])

    def plot_complete_bee_analysis(self, file_path, target_tags, duration_str=None):
        time_limit = None
        if duration_str:
            h, m, s = map(int, duration_str.split(':'))
            time_limit = timedelta(hours=h, minutes=m, seconds=s)

        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

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
        
        time_pat = TIME_PATTERN
        f_pat = F_VALUE_PATTERN
        sec_pat = re.compile(r'Possible feeding detected at secondary zone:\s+([\d.]+)')
        steps_pat = re.compile(r'number_of_steps set:\s+(\d+)')
        
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
                if s_match: current_steps_val = int(s_match.group(1))

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

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            subplot_titles=("Bee Triggers", "F-Sensor Binary (Normal vs Auto)", "F-Sensor Raw", "Secondary Area Activity", "Soft Stop Trigger"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
        )

        for tag, vals in bee_triggers.items():
            fig.add_trace(go.Scatter(x=times, y=vals, name=f"Bee {tag}", mode='lines'), row=1, col=1)

        fig.add_trace(go.Scatter(x=times, y=f_bin_normal, name="Normal Bump", line_shape='hv', line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=times, y=f_bin_autofeed, name="Auto-Feed Drinking", line_shape='hv', line=dict(color='orange'), fill='tozeroy'), row=2, col=1)

        fig.add_trace(go.Scatter(x=times, y=f_raw, name="Raw F-Value", line=dict(color='teal', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=times, y=[F_THRESHOLD]*len(times), name=f"Threshold ({F_THRESHOLD})", line=dict(color='red', dash='dash')), row=3, col=1)

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

        fig.add_trace(go.Scatter(x=times, y=soft_stop_binary, name="Soft Stop", line_shape='vh', line=dict(color='crimson')), row=5, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=times, y=steps_values, name="Steps Set", line_shape='hv', line=dict(color='blue', dash='dot')), row=5, col=1, secondary_y=True)

        fig.update_layout(height=1200, title_text=f"Interactive Bee Analysis", showlegend=True, hovermode="x unified")
        fig.update_xaxes(rangeslider_visible=True, row=5, col=1)

        output_name = file_path.replace(".txt", "_interactive.html")
        fig.write_html(output_name)
        self.log(f"Interactive graph saved to: {output_name}")

    def plot_scaled_feeding_cycles(self, file_path, target_tags):
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        self.log("Parsing log for sensor readings and feeding cycles...")
        
        sensor_data = [] # (time, b, m, f)
        sequences = [] # (t_eject, t_hold, t_retract, t_end)
        
        target_tags = [str(t) for t in target_tags]
        in_sequence = False
        current_phase = "IDLE"
        
        eject_start = hold_start = retract_start = retract_end = None
        
        # We need to compile the regex if not globally available, but SENSOR_LINE_PATTERN is.
        time_pat = TIME_PATTERN
        sensor_pat = SENSOR_LINE_PATTERN
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line.startswith('['): continue
                time_str = clean_line[1:20]
                try: current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError: continue
                
                # Check for sensors
                s_match = sensor_pat.search(clean_line)
                if s_match:
                    try:
                        b_val = int(s_match.group(1))
                        m_val = int(s_match.group(2))
                        f_val = int(s_match.group(3))
                        sensor_data.append((current_time, b_val, m_val, f_val))
                    except: pass
                
                # Check for feeding cycle
                is_motion_feed = "Tag None confirmed. Sending FEED command." in clean_line
                is_tag_feed = "confirmed. Sending FEED command." in clean_line and not is_motion_feed
                is_auto_feed = "AUTO FEED: Triggered" in clean_line
                
                if is_tag_feed or is_auto_feed or is_motion_feed:
                    detected_tag = None
                    if is_motion_feed:
                        detected_tag = "motion detection feeding"
                    elif is_tag_feed:
                        for t in target_tags:
                            if f"Tag {t} confirmed" in clean_line:
                                detected_tag = t
                                break
                    elif is_auto_feed:
                        detected_tag = "autofeed"
                        
                    if detected_tag and current_phase in ["IDLE", "RETRACTING"]:
                        if current_phase == "RETRACTING":
                            # Sequence ended early by another trigger, save previous one
                            if eject_start and hold_start and retract_start:
                                sequences.append((eject_start, hold_start, retract_start, current_time))
                        
                        in_sequence = True
                        eject_start = current_time
                        eject_end = hold_start = hold_end = retract_start = retract_end = None
                        current_phase = "EJECTING"
                    continue
                
                if "Starting HOLD" in clean_line and in_sequence:
                    hold_start = current_time
                    current_phase = "HOLDING"
                elif "Hold time over. Retracting." in clean_line and in_sequence:
                    retract_start = current_time
                    current_phase = "RETRACTING"
                elif "Retract Complete" in clean_line and in_sequence:
                    retract_end = current_time
                    if eject_start and hold_start and retract_start and retract_end:
                        sequences.append((eject_start, hold_start, retract_start, retract_end))
                    in_sequence = False
                    current_phase = "IDLE"
        
        self.log(f"Found {len(sequences)} complete feeding cycles. Extracting and scaling data...")
        if not sequences:
            self.log("No complete feeding cycles found to plot.")
            return
            
        all_scaled_f = []
        all_scaled_m = []
        all_scaled_b = []
        boundaries = []
        
        common_x = np.linspace(0, 1, 500)
        
        import bisect
        sensor_times = [s[0] for s in sensor_data]
        
        for (t_eject, t_hold, t_retract, t_end) in sequences:
            t_min = t_eject - timedelta(seconds=1)
            t_max = t_end + timedelta(seconds=1)
            duration = (t_max - t_min).total_seconds()
            if duration <= 0: continue
            
            start_idx = bisect.bisect_left(sensor_times, t_min)
            end_idx = bisect.bisect_right(sensor_times, t_max)
            
            seq_data = sensor_data[start_idx:end_idx]
            if not seq_data: continue
            
            t_vals = [(s[0] - t_min).total_seconds() / duration for s in seq_data]
            b_vals = [s[1] for s in seq_data]
            m_vals = [s[2] for s in seq_data]
            f_vals = [s[3] for s in seq_data]
            
            all_scaled_f.append(np.interp(common_x, t_vals, f_vals))
            all_scaled_m.append(np.interp(common_x, t_vals, m_vals))
            all_scaled_b.append(np.interp(common_x, t_vals, b_vals))
            
            b_eject = (t_eject - t_min).total_seconds() / duration
            b_hold = (t_hold - t_min).total_seconds() / duration
            b_retract = (t_retract - t_min).total_seconds() / duration
            b_end = (t_end - t_min).total_seconds() / duration
            boundaries.append((b_eject, b_hold, b_retract, b_end))
            
        if not all_scaled_f:
            self.log("No sensor data matched the feeding cycles.")
            return
            
        self.log("Plotting overlaid cycles...")
        fig, (ax_f, ax_m, ax_b) = plt.subplots(3, 1, figsize=(19.2, 10.8), dpi=100, sharex=True)
        
        for i in range(len(all_scaled_f)):
            ax_f.plot(common_x, all_scaled_f[i], color='gray', alpha=0.3, linewidth=1)
            ax_m.plot(common_x, all_scaled_m[i], color='gray', alpha=0.3, linewidth=1)
            ax_b.plot(common_x, all_scaled_b[i], color='gray', alpha=0.3, linewidth=1)
            
        mean_f = np.mean(all_scaled_f, axis=0)
        mean_m = np.mean(all_scaled_m, axis=0)
        mean_b = np.mean(all_scaled_b, axis=0)
        
        ax_f.plot(common_x, mean_f, color='black', linewidth=2, label='Average')
        ax_m.plot(common_x, mean_m, color='black', linewidth=2, label='Average')
        ax_b.plot(common_x, mean_b, color='black', linewidth=2, label='Average')
        
        avg_b_eject = np.mean([b[0] for b in boundaries])
        avg_b_hold = np.mean([b[1] for b in boundaries])
        avg_b_retract = np.mean([b[2] for b in boundaries])
        avg_b_end = np.mean([b[3] for b in boundaries])
        
        for ax in [ax_f, ax_m, ax_b]:
            for b_val in [avg_b_eject, avg_b_hold, avg_b_retract, avg_b_end]:
                ax.axvline(b_val, color='red', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
        ax_f.set_title("Front Sensor (F) - Scaled Cycle Time", fontsize=16)
        ax_m.set_title("Middle Sensor (M) - Scaled Cycle Time", fontsize=16)
        ax_b.set_title("Back Sensor (B) - Scaled Cycle Time", fontsize=16)
        
        ax_f.set_ylabel("Sensor Value")
        ax_m.set_ylabel("Sensor Value")
        ax_b.set_ylabel("Sensor Value")
        
        ax_b.set_xlabel("Scaled Cycle Time (0 to 1)", fontsize=14)
        
        # Add text labels at the bottom for the states
        trans = ax_b.get_xaxis_transform()
        y_pos = -0.15 # Below the x-axis
        
        ax_b.text((0 + avg_b_eject)/2, y_pos, "WAITING\n(Before)", transform=trans, ha='center', va='top', fontsize=12)
        ax_b.text((avg_b_eject + avg_b_hold)/2, y_pos, "EJECTING", transform=trans, ha='center', va='top', fontsize=12, fontweight='bold')
        ax_b.text((avg_b_hold + avg_b_retract)/2, y_pos, "HOLDING", transform=trans, ha='center', va='top', fontsize=12, fontweight='bold')
        ax_b.text((avg_b_retract + avg_b_end)/2, y_pos, "RETRACTING", transform=trans, ha='center', va='top', fontsize=12, fontweight='bold')
        ax_b.text((avg_b_end + 1.0)/2, y_pos, "WAITING\n(After)", transform=trans, ha='center', va='top', fontsize=12)
        
        plt.subplots_adjust(hspace=0.3, bottom=0.15)
        
        output_file = file_path.replace(".txt", "_scaled_cycles.png")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        
        self.log(f"Successfully plotted {len(sequences)} scaled cycles and saved to {output_file}")

    # Add remaining logic for drift & video extraction
    # Simplified here to keep file size manageable, but can easily import or implement.
    def analyze_baselines(self, file_path):
        data = {'WAITING': {'B': [], 'M': [], 'F': []}, 'HOLDING': {'B': [], 'M': [], 'F': []}}
        pattern = SENSOR_LINE_PATTERN
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

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

        self.log("=" * 50)
        self.log(f"{'SENSOR BASELINE ANALYSIS':^50}")
        self.log("=" * 50)
        for state in ['WAITING', 'HOLDING']:
            self.log(f"\n>>> STATE: {state} <<<")
            for sensor in ['B', 'M', 'F']:
                values = data[state][sensor]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    self.log(f"  Sensor {sensor}: Avg={mean_val:>7.2f} | Std={std_val:>5.2f} | Range=[{min_val}, {max_val}]")
                else:
                    self.log(f"  Sensor {sensor}: No data found.")
        self.log("\n" + "=" * 50)



    def analyze_possible_feeding(self, file_path, output_csv, min_duration=0.0, max_duration=float('inf'), min_avg_intensity=0.0, merge_gap=8.0):
        target_phrase = "Possible feeding detected at secondary zone:"
        
        current_chain_times = []
        current_chain_values = []
        last_processed_time = None
        last_valid_end_dt = None
        chain_data_for_csv = []
        
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        with open(file_path, 'r') as file:
            for line in file:
                clean_line = line.strip()
                if not clean_line.startswith('['):
                    continue
                    
                try:
                    time_str = clean_line[1:20]
                    current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                    
                    if target_phrase in clean_line:
                        val_str = clean_line.split(":")[-1].strip()
                        val = float(val_str)
                        
                        if last_processed_time and (current_time - last_processed_time).total_seconds() <= max(1.0, merge_gap):
                            current_chain_times.append(current_time)
                            current_chain_values.append(val)
                        else:
                            if current_chain_values:
                                start_dt = current_chain_times[0]
                                end_dt = current_chain_times[-1]
                                duration = (end_dt - start_dt).total_seconds()
                                
                                if min_duration <= duration <= max_duration:
                                    mn, mx = min(current_chain_values), max(current_chain_values)
                                    avg = sum(current_chain_values) / len(current_chain_values)
                                    
                                    if avg >= min_avg_intensity:
                                        time_since_last = ""
                                        if last_valid_end_dt:
                                            time_since_last = round((start_dt - last_valid_end_dt).total_seconds(), 2)
                                        last_valid_end_dt = end_dt
                                        
                                        chain_data_for_csv.append([
                                            start_dt.strftime("%Y-%m-%d %H:%M:%S"), 
                                            end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                            mn, mx, round(avg, 2), len(current_chain_values), duration, time_since_last
                                        ])

                            current_chain_times = [current_time]
                            current_chain_values = [val]
                        
                        last_processed_time = current_time
                except (ValueError, IndexError):
                    continue

        if current_chain_values:
            start_dt = current_chain_times[0]
            end_dt = current_chain_times[-1]
            duration = (end_dt - start_dt).total_seconds()
            if min_duration <= duration <= max_duration:
                mn, mx = min(current_chain_values), max(current_chain_values)
                avg = sum(current_chain_values) / len(current_chain_values)
                
                if avg >= min_avg_intensity:
                    time_since_last = ""
                    if last_valid_end_dt:
                        time_since_last = round((start_dt - last_valid_end_dt).total_seconds(), 2)
                    last_valid_end_dt = end_dt
                    
                    chain_data_for_csv.append([
                        start_dt.strftime("%Y-%m-%d %H:%M:%S"), 
                        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        mn, mx, round(avg, 2), len(current_chain_values), duration, time_since_last
                    ])

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Start Timestamp', 'End Timestamp', 'Min Intensity', 'Max Intensity', 'Avg Intensity', 'Log Count', 'Duration (s)', 'Time Since Last (s)'])
            writer.writerows(chain_data_for_csv)

        self.log(f"Analysis complete. Found {len(chain_data_for_csv)} distinct feeding chains saved to {output_csv}.")

    def extract_possible_feeding_clips(self, sequence_csv, log_file, video_file, output_folder="possible feedings", pre_pad_sec=2.0, post_pad_sec=3.0, min_duration=0.0, max_duration=float('inf'), min_avg_intensity=0.0):
        if not os.path.exists(log_file) or not os.path.exists(video_file) or not os.path.exists(sequence_csv):
            self.log("Error: One or more input files not found for video extraction.")
            return

        os.makedirs(output_folder, exist_ok=True)
        recording_start_time = None
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Recording started:" in line and ".mp4" in line:
                    m = TIME_PATTERN.search(line)
                    if m:
                        recording_start_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        break
        
        if not recording_start_time:
            self.log("ERROR: Could not find 'Recording started:' line in log.")
            return

        self.log(f"Recording start time: {recording_start_time}")
        total_rows = saved_clips = skipped_rows = 0
        summary_path = os.path.join(output_folder, "possible_feeding_extraction_summary.csv")

        with open(sequence_csv, "r", newline="", encoding="utf-8", errors="ignore") as infile, \
             open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
            reader = csv.DictReader(infile)
            writer = csv.writer(summary_file)
            writer.writerow(["Clip Index", "Start Timestamp", "Video Start (s)", "Clip Duration (s)", "Status"])

            for row in reader:
                total_rows += 1
                try:
                    tag_time_str = row["Start Timestamp"]
                    tag_time = datetime.strptime(tag_time_str, "%Y-%m-%d %H:%M:%S")
                    sequence_duration = float(row["Duration (s)"])
                    avg_intensity = float(row.get("Avg Intensity", 0.0))
                except Exception as e:
                    skipped_rows += 1
                    self.log(f"Skipping row {total_rows} due to parse error: {e}")
                    continue

                if sequence_duration < min_duration or sequence_duration > max_duration or avg_intensity < min_avg_intensity:
                    skipped_rows += 1
                    self.log(f"Skipping row {total_rows} due to filter (duration={sequence_duration}s, avg_intensity={avg_intensity})")
                    continue

                raw_video_start = (tag_time - recording_start_time).total_seconds()
                actual_pre_pad = min(pre_pad_sec, max(0.0, raw_video_start))
                video_start = max(0.0, raw_video_start - pre_pad_sec)
                clip_duration = actual_pre_pad + sequence_duration + post_pad_sec

                if clip_duration <= 0:
                    skipped_rows += 1
                    continue

                clip_name = f"possible_feeding_{total_rows:04d}.mp4"
                output_path = os.path.join(output_folder, clip_name)

                command = ["ffmpeg", "-y", "-ss", f"{video_start:.3f}", "-i", video_file, "-t", f"{clip_duration:.3f}", "-c", "copy", output_path]
                try:
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0 and os.path.exists(output_path):
                        saved_clips += 1
                        self.log(f"Saved {clip_name} | start={video_start:.2f}s | duration={clip_duration:.2f}s")
                        writer.writerow([total_rows, tag_time_str, round(video_start, 3), round(clip_duration, 3), "Saved"])
                    else:
                        skipped_rows += 1
                        self.log(f"ERROR cutting {clip_name}")
                except Exception as e:
                    skipped_rows += 1
                    self.log(f"Exception extracting {clip_name}: {e}")

        self.log(f"Done! Saved {saved_clips} possible feeding clips. Skipped {skipped_rows} rows.")

    def downsize_video(self, video_path, target_mb):
        if not os.path.exists(video_path):
            self.log(f"Error: Video file {video_path} not found.")
            return

        self.log(f"Starting downsize process for {os.path.basename(video_path)} to {target_mb} MB")
        
        # Get duration
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            duration = float(result.stdout.strip())
            self.log(f"Video duration: {duration:.2f} seconds")
        except Exception as e:
            self.log(f"Error getting video duration: {e}")
            return

        # Calculate bitrates
        # Target size in bits = MB * 1024 * 1024 * 8
        target_total_bitrate = (target_mb * 1024 * 1024 * 8) / duration
        
        # Audio bitrate (assume 128k if there is room, else lower or 0)
        audio_bitrate = 128000
        if target_total_bitrate <= audio_bitrate:
            audio_bitrate = 64000
            
        video_bitrate = target_total_bitrate - audio_bitrate
        if video_bitrate <= 0:
            self.log("Error: Target size is too small for this video duration.")
            return

        self.log(f"Target Video Bitrate: {int(video_bitrate/1000)}k, Audio: {int(audio_bitrate/1000)}k")

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(os.path.dirname(video_path), f"{base_name}_downsized.mp4")

        # 2-pass encoding
        self.log("Starting Pass 1...")
        # pass 1 does not need audio
        pass1_cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-b:v", str(int(video_bitrate)), "-pass", "1", "-an", "-f", "null", "NUL" if os.name == 'nt' else "/dev/null"]
        
        try:
            subprocess.run(pass1_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            self.log(f"Error in Pass 1: {e.stderr.decode('utf-8', errors='ignore')}")
            return

        self.log("Starting Pass 2...")
        pass2_cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-b:v", str(int(video_bitrate)), "-pass", "2", "-c:a", "aac", "-b:a", str(int(audio_bitrate)), output_file]
        
        try:
            subprocess.run(pass2_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.log(f"Successfully downsized video to {output_file}")
            # Verify file size
            actual_size = os.path.getsize(output_file) / (1024 * 1024)
            self.log(f"Actual size: {actual_size:.2f} MB")
        except subprocess.CalledProcessError as e:
            self.log(f"Error in Pass 2: {e.stderr.decode('utf-8', errors='ignore')}")
            return

    def plot_possible_feedings(self, csv_file):
        if not os.path.exists(csv_file):
            self.log(f"Error: {csv_file} not found.")
            return

        times = []
        durations = []
        
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(datetime.strptime(row['Start Timestamp'], "%Y-%m-%d %H:%M:%S"))
                    durations.append(float(row['Duration (s)']))
        except Exception as e:
            self.log(f"Error reading CSV: {e}")
            return
            
        if not times:
            self.log("No data found in CSV to plot.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=durations,
            mode='markers',
            marker=dict(size=10, color='#f9c901', opacity=0.8, line=dict(width=1, color='white')),
            name="Feeding Duration",
            hovertemplate="Time: %{x}<br>Duration: %{y} s<extra></extra>"
        ))

        fig.update_layout(
            title="Possible Feedings Duration over Time",
            xaxis_title="Time of Day",
            yaxis_title="Duration (Seconds)",
            template="plotly_dark",
            hovermode="closest",
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='#FFF')
        )
        
        base_name = os.path.splitext(os.path.basename(csv_file))[0].replace("_possible_feedings", "")
        out_html = os.path.join(os.path.dirname(csv_file), f"{base_name}_possible_feedings_plot.html")
        
        try:
            fig.write_html(out_html)
            self.log(f"Successfully generated interactive scatter plot: {out_html}")
        except Exception as e:
            self.log(f"Error saving plot: {e}")

    def plot_all_electrodes(self, file_path, time_limit_str=None):
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        time_limit = None
        if time_limit_str:
            try:
                h, m, s = map(int, time_limit_str.split(':'))
                time_limit = timedelta(hours=h, minutes=m, seconds=s)
            except Exception:
                pass

        times = []
        b_vals = []
        m_vals = []
        f_vals = []
        
        start_time = None
        time_pat = TIME_PATTERN
        sensor_pat = SENSOR_LINE_PATTERN

        self.log(f"Parsing {os.path.basename(file_path)} for electrode values...")
        try:
            with open(file_path, 'r', encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    t_match = time_pat.search(line)
                    if not t_match: continue
                    
                    curr_time = datetime.strptime(t_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time is None: start_time = curr_time
                    if time_limit and (curr_time - start_time) > time_limit: break

                    if "SENSORS:" in line:
                        s_match = sensor_pat.search(line)
                        if s_match:
                            b_val = int(s_match.group(1))
                            m_val = int(s_match.group(2))
                            f_val = int(s_match.group(3))
                            
                            times.append(curr_time)
                            b_vals.append(b_val)
                            m_vals.append(m_val)
                            f_vals.append(f_val)
        except Exception as e:
            self.log(f"Error parsing log file: {e}")
            return
            
        if not times:
            self.log("No electrode data found in log file.")
            return

        self.log("Generating plot...")
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=("Back Electrode (B)", "Middle Electrode (M)", "Front Electrode (F)")
        )

        fig.add_trace(go.Scatter(x=times, y=b_vals, name='Back (B)', line=dict(color='#ff9900')), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=m_vals, name='Middle (M)', line=dict(color='#00ccff')), row=2, col=1)
        fig.add_trace(go.Scatter(x=times, y=f_vals, name='Front (F)', line=dict(color='#ff3366')), row=3, col=1)

        fig.update_layout(
            title="Electrode Values over Time",
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='#FFF'),
            height=800
        )
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_html = os.path.join(os.path.dirname(file_path), f"{base_name}_electrodes_plot.html")
        
        try:
            fig.write_html(out_html)
            self.log(f"Successfully generated interactive electrode plot: {out_html}")
        except Exception as e:
            self.log(f"Error saving plot: {e}")

    def plot_soft_stop(self, file_path, time_limit_str=None):
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        time_limit = None
        if time_limit_str:
            try:
                h, m, s = map(int, time_limit_str.split(':'))
                time_limit = timedelta(hours=h, minutes=m, seconds=s)
            except Exception:
                pass

        times = []
        soft_stop_binary = []
        
        start_time = None
        time_pat = TIME_PATTERN

        self.log(f"Parsing {os.path.basename(file_path)} for Soft Stop triggers...")
        try:
            with open(file_path, 'r', encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    t_match = time_pat.search(line)
                    if not t_match: continue
                    
                    curr_time = datetime.strptime(t_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time is None: start_time = curr_time
                    if time_limit and (curr_time - start_time) > time_limit: break

                    is_soft_stop = "STATUS: Soft Stop (Distance). Starting HOLD." in line
                    times.append(curr_time)
                    soft_stop_binary.append(1 if is_soft_stop else 0)
        except Exception as e:
            self.log(f"Error parsing log file: {e}")
            return
            
        if not times:
            self.log("No timestamped data found in log file.")
            return

        self.log("Generating plot...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=soft_stop_binary, name='Soft Stop', mode='lines', line_shape='hv', line=dict(color='crimson')))

        fig.update_layout(
            title="Soft Stop Triggers over Time",
            xaxis_title="Time",
            yaxis_title="Triggered (1) / Not Triggered (0)",
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='#FFF')
        )
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_html = os.path.join(os.path.dirname(file_path), f"{base_name}_soft_stop_plot.html")
        
        try:
            fig.write_html(out_html)
            self.log(f"Successfully generated interactive soft stop plot: {out_html}")
        except Exception as e:
            self.log(f"Error saving plot: {e}")

    def plot_steps_set(self, file_path, time_limit_str=None):
        if not os.path.exists(file_path):
            self.log(f"Error: {file_path} not found.")
            return

        time_limit = None
        if time_limit_str:
            try:
                h, m, s = map(int, time_limit_str.split(':'))
                time_limit = timedelta(hours=h, minutes=m, seconds=s)
            except Exception:
                pass

        times = []
        steps_values = []
        
        start_time = None
        time_pat = TIME_PATTERN
        steps_pat = re.compile(r'number_of_steps set:\s+(\d+)')
        
        current_steps_val = 0

        self.log(f"Parsing {os.path.basename(file_path)} for number of steps set...")
        try:
            with open(file_path, 'r', encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    t_match = time_pat.search(line)
                    if not t_match: continue
                    
                    curr_time = datetime.strptime(t_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time is None: start_time = curr_time
                    if time_limit and (curr_time - start_time) > time_limit: break

                    s_match = steps_pat.search(line)
                    if s_match:
                        current_steps_val = int(s_match.group(1))

                    times.append(curr_time)
                    steps_values.append(current_steps_val)
        except Exception as e:
            self.log(f"Error parsing log file: {e}")
            return
            
        if not times:
            self.log("No timestamped data found in log file.")
            return

        self.log("Generating plot...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=steps_values, name='Steps Set', mode='lines', line_shape='hv', line=dict(color='blue')))

        fig.update_layout(
            title="Number of Steps Set over Time",
            xaxis_title="Time",
            yaxis_title="Number of Steps",
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='#FFF')
        )
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_html = os.path.join(os.path.dirname(file_path), f"{base_name}_steps_set_plot.html")
        
        try:
            fig.write_html(out_html)
            self.log(f"Successfully generated interactive steps plot: {out_html}")
        except Exception as e:
            self.log(f"Error saving plot: {e}")
