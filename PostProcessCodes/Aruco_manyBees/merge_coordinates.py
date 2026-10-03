import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import datetime

def merge_coordinate_files(file_list, output_file):
    """
    Merges a list of coordinate files into a single output file.
    Preserves the header from the first file and skips headers in subsequent files.
    """
    if not file_list:
        return
        
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, filename in enumerate(file_list):
            if not os.path.exists(filename):
                continue
                
            with open(filename, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                
                if not lines:
                    continue
                    
                # If it's the first file, write the header, otherwise skip the first line
                if i == 0:
                    outfile.write(lines[0])
                
                # Write the rest of the lines (skipping the header for all files)
                outfile.writelines(lines[1:])

def merge_info_files(file_list, output_file):
    """
    Merges info.txt files into a single file with cumulative/list settings.
    """
    if not file_list:
        return

    # Keys to accumulate
    cumulative = ["end_time_s", "end_frame", "planned_total_frames"]
    # Keys to list
    list_settings = ["video_path", "single_bee_settings_path"]
    
    merged_lines = []
    
    # We will compute sums here
    sums = { "end_time_s": 0.0, "end_frame": 0, "planned_total_frames": 0 }
    lists = { "video_path": [], "single_bee_settings_path": [] }
    
    # Pass 1: get sums and lists from all files
    for filename in file_list:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if key in cumulative:
                            if key == "end_time_s":
                                sums[key] += float(val)
                            else:
                                sums[key] += int(float(val))
                        elif key in list_settings:
                            lists[key].append(val)
        except Exception as e:
            print(f"Warning: could not read {filename} during pass 1: {e}")

    # Pass 2: read first file, replace values, write to out
    try:
        with open(file_list[0], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts:
                    merged_lines.append(line)
                    continue
                    
                key = parts[0].strip()
                if key in cumulative:
                    merged_lines.append(f"{key}\t{sums[key]}\n")
                elif key in list_settings:
                    merged_lines.append(f"{key}\t{', '.join(lists[key])}\n")
                elif key == "end_time_str":
                    end_seconds = int(sums["end_time_s"])
                    end_str = str(datetime.timedelta(seconds=end_seconds))
                    if len(end_str.split(':')) == 3 and len(end_str.split(':')[0]) == 1:
                        end_str = '0' + end_str
                    merged_lines.append(f"{key}\t{end_str}\n")
                else:
                    merged_lines.append(line)
    except Exception as e:
        print(f"Error reading first file during pass 2: {e}")
        return
                
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(merged_lines)


class MergeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coordinate Merger")
        self.root.geometry("500x400")
        
        self.input_folders = []
        self.output_folder = ""
        
        # UI Elements
        tk.Label(root, text="Input Folders (Ordered):").pack(pady=(10,0))
        
        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=60, height=8)
        self.listbox.pack(pady=5)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Remove Selected", command=self.remove_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Move Up", command=lambda: self.move_folder(-1)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Move Down", command=lambda: self.move_folder(1)).pack(side=tk.LEFT, padx=5)
        
        tk.Label(root, text="Output Folder:").pack(pady=(15,0))
        
        self.out_frame = tk.Frame(root)
        self.out_frame.pack(pady=5)
        self.out_label = tk.Label(self.out_frame, text="Not selected", width=45, anchor='w', bg="white", relief="sunken")
        self.out_label.pack(side=tk.LEFT, padx=5)
        tk.Button(self.out_frame, text="Browse", command=self.select_output).pack(side=tk.LEFT)
        
        tk.Button(root, text="Merge Files", command=self.merge_files, bg="lightblue", font=("Arial", 10, "bold")).pack(pady=20)
        
    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Folder containing coordinates")
        if folder:
            self.input_folders.append(folder)
            self.update_listbox()
            
    def remove_folder(self):
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.input_folders.pop(idx)
            self.update_listbox()
            
    def move_folder(self, direction):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        new_idx = idx + direction
        if 0 <= new_idx < len(self.input_folders):
            self.input_folders[idx], self.input_folders[new_idx] = self.input_folders[new_idx], self.input_folders[idx]
            self.update_listbox()
            self.listbox.selection_set(new_idx)
            
    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for folder in self.input_folders:
            self.listbox.insert(tk.END, folder)
            
    def select_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.out_label.config(text=folder)
            
    def merge_files(self):
        if not self.input_folders:
            messagebox.showerror("Error", "Please add at least one input folder.")
            return
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return
            
        raw_files = []
        filtered_files = []
        info_files = []
        
        for folder in self.input_folders:
            # Recursively search for raw, filtered, and info txt files in the given folder
            found_raw = glob.glob(os.path.join(folder, "**", "*_coordinates_raw.txt"), recursive=True)
            found_filtered = glob.glob(os.path.join(folder, "**", "*_coordinates_filtered.txt"), recursive=True)
            found_info = glob.glob(os.path.join(folder, "**", "*_info.txt"), recursive=True)
            
            if not found_raw or not found_filtered:
                messagebox.showwarning("Warning", f"Could not find both raw and filtered coordinate files in:\n{folder}\nCheck subfolders.")
                continue
                
            raw_files.append(found_raw[0])
            filtered_files.append(found_filtered[0])
            if found_info:
                info_files.append(found_info[0])
            
        if not raw_files or not filtered_files:
            messagebox.showerror("Error", "No valid coordinate files found to merge.")
            return
            
        try:
            merged_raw_path = os.path.join(self.output_folder, "merged_coordinates_raw.txt")
            merged_filtered_path = os.path.join(self.output_folder, "merged_coordinates_filtered.txt")
            merged_info_path = os.path.join(self.output_folder, "merged_info.txt")
            
            merge_coordinate_files(raw_files, merged_raw_path)
            merge_coordinate_files(filtered_files, merged_filtered_path)
            if info_files:
                merge_info_files(info_files, merged_info_path)
            
            msg = f"Successfully merged coordinates into:\n{self.output_folder}"
            if info_files:
                msg += "\n(Merged info.txt as well)"
            messagebox.showinfo("Success", msg)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while merging:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MergeApp(root)
    root.mainloop()
