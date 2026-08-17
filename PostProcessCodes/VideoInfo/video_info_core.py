import subprocess
import json
import os
from typing import Dict, Any, Optional

class VideoInfoExtractor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.raw_data: Optional[Dict[str, Any]] = None

    def extract_info(self) -> Dict[str, Any]:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            self.video_path
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.raw_data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed. Error: {e.stderr}")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse ffprobe output as JSON.")
        except FileNotFoundError:
            raise RuntimeError("ffprobe not found. Please ensure ffmpeg is installed and added to your PATH.")

        return self._parse_raw_data(self.raw_data)

    def _parse_raw_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        parsed = {
            "general": {},
            "video": {},
            "audio": {}
        }
        
        # General / Format
        fmt = data.get("format", {})
        parsed["general"]["filename"] = os.path.basename(self.video_path)
        size_bytes = int(fmt.get("size", 0))
        parsed["general"]["size_mb"] = round(size_bytes / (1024 * 1024), 2)
        parsed["general"]["duration_sec"] = float(fmt.get("duration", 0))
        parsed["general"]["overall_bitrate_kbps"] = round(float(fmt.get("bit_rate", 0)) / 1000, 2) if fmt.get("bit_rate") else 0
        parsed["general"]["container_format"] = fmt.get("format_long_name", "Unknown")

        # Streams
        streams = data.get("streams", [])
        for stream in streams:
            codec_type = stream.get("codec_type")
            if codec_type == "video" and not parsed["video"]:
                parsed["video"]["codec"] = stream.get("codec_name", "Unknown")
                parsed["video"]["profile"] = stream.get("profile", "Unknown")
                parsed["video"]["width"] = stream.get("width", 0)
                parsed["video"]["height"] = stream.get("height", 0)
                parsed["video"]["pixel_format"] = stream.get("pix_fmt", "Unknown")
                parsed["video"]["display_aspect_ratio"] = stream.get("display_aspect_ratio", "Unknown")
                parsed["video"]["bitrate_kbps"] = round(float(stream.get("bit_rate", 0)) / 1000, 2) if stream.get("bit_rate") else 0
                
                # frame rate logic e.g. "30/1"
                fps_str = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) > 0 else 0
                except ValueError:
                    fps = 0
                parsed["video"]["fps"] = round(fps, 3)

                parsed["video"]["total_frames"] = stream.get("nb_frames", "Unknown")

            elif codec_type == "audio" and not parsed["audio"]:
                parsed["audio"]["codec"] = stream.get("codec_name", "Unknown")
                parsed["audio"]["sample_rate_hz"] = stream.get("sample_rate", "Unknown")
                parsed["audio"]["channels"] = stream.get("channels", "Unknown")
                parsed["audio"]["bitrate_kbps"] = round(float(stream.get("bit_rate", 0)) / 1000, 2) if stream.get("bit_rate") else 0
                parsed["audio"]["channel_layout"] = stream.get("channel_layout", "Unknown")
                
        return parsed
