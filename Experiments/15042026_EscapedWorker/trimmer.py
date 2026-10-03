import subprocess
from pathlib import Path

def trim_mp4_fast(input_file, start_time, end_time, output_file):
    """
    Trim an MP4 between start_time and end_time using ffmpeg stream copy.

    Parameters:
        input_file (str): Path to input mp4
        start_time (str): Start time, e.g. "00:01:15.500"
        end_time   (str): End time, e.g. "00:02:10.000"
        output_file(str): Path to output mp4
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_time,
        "-to", end_time,
        "-i", str(input_path),
        "-c", "copy",
        str(output_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"Trimmed video saved to: {output_file}")


if __name__ == "__main__":
    trim_mp4_fast(
        input_file="feed_output_20260415_133807_1920x1080.mp4",
        start_time="00:00:00",
        end_time="00:15:33",
        output_file="trimmed.mp4"
    )