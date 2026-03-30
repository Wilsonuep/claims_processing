import os
import time
from datetime import datetime

FILE_PATH = "polish_wikipedia_articles.jsonl"
INTERVAL_SECONDS = 600  # 10 minutes


def get_file_size(path):
    """Returns file size in GB using OS metadata only (no file reading)."""
    size_bytes = os.path.getsize(path)
    size_gb = size_bytes / (1024 ** 3)
    return size_bytes, size_gb

def monitor_file(path, interval=60):
    prev_size = None

    while True:
        if not os.path.exists(path):
            print(f"{datetime.now().isoformat()} | File does not exist yet: {path}")
            time.sleep(5)
            continue

        size_bytes, size_gb = get_file_size(path)
        ts = datetime.now().isoformat(timespec="seconds")

        if prev_size is not None:
            delta_mb = (size_bytes - prev_size) / (1024 ** 2)
            print(f"{ts} | size: {size_gb:.6f} GB ({size_bytes:,} B) | delta: +{delta_mb:.2f} MB")
        else:
            print(f"{ts} | size: {size_gb:.6f} GB ({size_bytes:,} B)")

        # stop if not growing
        if prev_size is not None and size_bytes == prev_size:
            print(f"{ts} | File seems to have stopped growing. Waiting 5s to re-check...")
            time.sleep(5)
            size_bytes, size_gb = get_file_size(path)  
            if size_bytes == prev_size:
                print(f"{datetime.now().isoformat(timespec='seconds')} | Still no growth after 5s. Exiting.")
                break
            else:
                delta_mb = (size_bytes - prev_size) / (1024 ** 2)
                print(f"{datetime.now().isoformat(timespec='seconds')} | Growth detected! delta: +{delta_mb:.2f} MB")

        prev_size = size_bytes
        time.sleep(interval)

if __name__ == "__main__":
    monitor_file(FILE_PATH, INTERVAL_SECONDS)