import csv
import os
from datetime import datetime

"""
logger.py — Trip CSV Logger

Responsibility: Save every raw sensor reading that arrives at the server
to a CSV file on disk, one row per reading, one file per trip.

The CSV files written here serve two purposes:
  1. A permanent record of every live trip for auditing and debugging.
  2. The source file that replay.py reads back during demo day
     (demo mode replays a pre-recorded CSV instead of needing a real car).

Each trip gets its own file named:
  trip_<session_id>_<timestamp>.csv
saved under server/data/logs/.

Nothing in this file does any scoring or processing — it just writes
raw data to disk as fast and safely as possible.
"""

# Where all trip CSV files will be saved on disk.
# os.path.dirname(__file__) means "the directory this file lives in",
# so this resolves to server/data/logs/ regardless of where you run the script from.
LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "logs")

# The exact columns we write to every CSV row, in order.
# These match the flattened fields from the JSON contract.
# Keeping them here as a constant means the writer and any reader
# (like replay.py later) can import this list instead of hardcoding it twice.
COLUMNS = [
    "ts", "session_id",
    "speed", "rpm", "throttle", "engine_load", "maf",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "lat", "lon", "speed_gps", "accuracy"
]


def get_log_path(session_id: str) -> str:
    """
    Build the full file path for a new trip log CSV.

    Creates the logs directory if it doesn't exist yet.
    The filename includes both the session_id and the current timestamp
    so that two trips with the same session_id (shouldn't happen, but just
    in case) don't overwrite each other.

    Example output: server/data/logs/trip_abc123_20260123_210749.csv
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trip_{session_id}_{date_str}.csv"
    return os.path.join(LOG_DIR, filename)


def flatten_reading(reading: dict) -> dict:
    """
    Convert one nested JSON reading into a flat dict suitable for a CSV row.

    The incoming JSON has nested sections (obd, imu, gps). CSV rows are flat.
    This function unpacks them into a single-level dict with one key per column.

    .get("obd", {}).get("speed") means:
      - get the "obd" section from the reading, or an empty dict if it's missing
      - then get "speed" from that, or None if it's missing
    Using None for missing values is intentional — it writes as an empty cell
    in the CSV, which is distinguishable from a genuine zero.
    """
    return {
        "ts":          reading.get("ts"),
        "session_id":  reading.get("session_id"),
        # OBD fields from the vehicle computer
        "speed":       reading.get("obd", {}).get("speed"),
        "rpm":         reading.get("obd", {}).get("rpm"),
        "throttle":    reading.get("obd", {}).get("throttle"),
        "engine_load": reading.get("obd", {}).get("engine_load"),
        "maf":         reading.get("obd", {}).get("maf"),
        # IMU fields from the phone's accelerometer and gyroscope
        "accel_x":     reading.get("imu", {}).get("accel_x"),
        "accel_y":     reading.get("imu", {}).get("accel_y"),
        "accel_z":     reading.get("imu", {}).get("accel_z"),
        "gyro_x":      reading.get("imu", {}).get("gyro_x"),
        "gyro_y":      reading.get("imu", {}).get("gyro_y"),
        "gyro_z":      reading.get("imu", {}).get("gyro_z"),
        # GPS fields from the phone's location provider
        "lat":         reading.get("gps", {}).get("lat"),
        "lon":         reading.get("gps", {}).get("lon"),
        "speed_gps":   reading.get("gps", {}).get("speed_gps"),
        "accuracy":    reading.get("gps", {}).get("accuracy"),
    }


class TripLogger:
    """
    Manages the CSV log file for one trip (one session).

    One TripLogger instance is created when a new session_id is seen
    by the server. It stays open and keeps appending rows until the
    trip ends or the server shuts down.

    Usage:
        logger = TripLogger("abc123")
        logger.log(reading_dict)   # call once per incoming reading
        logger.close()             # call when the trip ends
    """

    def __init__(self, session_id: str):
        """
        Open a new CSV file for this session and write the header row.

        Args:
            session_id: The trip identifier sent by the Android app.
        """
        self.session_id = session_id
        self.path = get_log_path(session_id)

        # Open the file in write mode. newline="" is required by Python's
        # csv module on all platforms to avoid double newlines on Windows.
        self._file = open(self.path, "w", newline="")

        # DictWriter lets us write dicts directly as rows,
        # matching keys to columns automatically.
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()

    def log(self, reading: dict):
        """
        Flatten one raw JSON reading and append it as a CSV row.

        Also flushes immediately so the row is on disk even if the
        server crashes before the next write. This is important because
        the CSV is our source of truth for demo replay mode.

        Args:
            reading: The full JSON dict as received from the Android app.
        """
        row = flatten_reading(reading)
        self._writer.writerow(row)
        # Force write to disk immediately — don't wait for Python's buffer.
        self._file.flush()

    def close(self):
        """
        Close the CSV file cleanly.

        Call this when a trip ends or the server shuts down.
        Not closing the file is not catastrophic (flush() keeps data safe)
        but it's good practice.
        """
        self._file.close()