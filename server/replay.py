"""
replay.py — Demo Day CSV Replay Engine

Responsibility: Read a pre-recorded trip CSV and replay it row by row
at real speed in a background thread, feeding each row through the
same buffer and scorer used in live mode.

Why this exists:
  On demo day we cannot guarantee a car, a parking spot, good WiFi,
  or a working OBD connection. Replay mode lets us demonstrate the
  full scoring pipeline — including real score escalation when Driver G
  drives — using a CSV we recorded during a real test drive.

How it works:
  1. ReplayEngine is started with the path to a trip CSV.
  2. A background thread reads rows one by one.
  3. Between rows it sleeps for the same time gap as the original
     recording — so a 22-minute trip replays in 22 real minutes.
     Pass speed_multiplier > 1 to replay faster (e.g. 2.0 = 2x speed).
  4. Each row is reconstructed into the nested JSON format that
     buffer.py and scorer.py expect, then processed identically
     to a live reading.
  5. The latest score dict is stored in latest_result, which
     GET /latest_score reads directly.

Usage:
    engine = ReplayEngine("data/logs/trip_abc123_20260123.csv")
    engine.start()
    # later:
    result = engine.latest_result   # None until first window is scored
    engine.stop()
"""

import csv
import time
import threading
from datetime import datetime

from buffer import SensorBuffer
from scorer import TripScorer
from logger import COLUMNS


def _row_to_reading(row):
    """
    Reconstruct a flat CSV row dict back into the nested JSON format
    that buffer.py and scorer.py expect.

    This is the inverse of logger.flatten_reading(). We need to undo
    the flattening because SensorBuffer.add() and TripScorer.score_window()
    were designed to work with the nested format.

    Args:
        row: A dict from csv.DictReader — all values are strings.

    Returns:
        A nested dict matching the JSON contract (obd/imu/gps sections).
        Numeric strings are converted to float. Empty strings become None.
    """
    def to_float(val):
        # Convert string to float, or None if empty/missing.
        try:
            return float(val) if val != "" else None
        except (ValueError, TypeError):
            return None

    return {
        "ts":         row.get("ts"),
        "session_id": row.get("session_id"),
        "obd": {
            "speed":        to_float(row.get("speed")),
            "rpm":          to_float(row.get("rpm")),
            "throttle":     to_float(row.get("throttle")),
            "engine_load":  to_float(row.get("engine_load")),
            "maf":          to_float(row.get("maf")),
        },
        "imu": {
            "accel_x": to_float(row.get("accel_x")),
            "accel_y": to_float(row.get("accel_y")),
            "accel_z": to_float(row.get("accel_z")),
            "gyro_x":  to_float(row.get("gyro_x")),
            "gyro_y":  to_float(row.get("gyro_y")),
            "gyro_z":  to_float(row.get("gyro_z")),
        },
        "gps": {
            "lat":       to_float(row.get("lat")),
            "lon":       to_float(row.get("lon")),
            "speed_gps": to_float(row.get("speed_gps")),
            "accuracy":  to_float(row.get("accuracy")),
        },
    }


class ReplayEngine:
    """
    Replays a recorded trip CSV through the scoring pipeline.

    Thread-safe: latest_result can be read from the main thread
    while the replay thread is running.
    """

    def __init__(self, csv_path, speed_multiplier=1.0):
        """
        Args:
            csv_path:         Path to the trip CSV file to replay.
            speed_multiplier: How fast to replay relative to real time.
                              1.0 = real time, 2.0 = twice as fast.
                              Useful for demos where you want to show
                              score escalation without waiting 22 minutes.
        """
        self.csv_path         = csv_path
        self.speed_multiplier = speed_multiplier

        # The most recent score dict produced by the scorer.
        # None until the first full 30-second window has been processed.
        # Written by the replay thread, read by GET /latest_score.
        self.latest_result = None

        # Internal state.
        self._buffer  = SensorBuffer()
        self._scorer  = TripScorer()
        self._thread  = None
        self._running = False

    def start(self):
        """
        Start the replay in a background thread.

        Returns immediately — replay runs concurrently with the Flask server.
        """
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Replay started: {self.csv_path} at {self.speed_multiplier}x speed")

    def stop(self):
        """Stop the replay thread cleanly."""
        self._running = False

    def _run(self):
        """
        Main replay loop — runs in the background thread.

        Reads CSV rows one by one, sleeps between them to match the
        original recording's timing, and feeds each row to the buffer
        and scorer exactly as live mode would.
        """
        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows   = list(reader)

            if not rows:
                print("Replay: CSV file is empty.")
                return

            print(f"Replay: loaded {len(rows)} rows.")
            prev_ts = None

            for row in rows:
                if not self._running:
                    break

                # Parse the timestamp so we can compute the sleep duration.
                try:
                    ts = datetime.fromisoformat(row.get("ts", ""))
                except ValueError:
                    # Malformed timestamp — skip this row.
                    continue

                # Sleep for the same gap as in the original recording,
                # scaled by speed_multiplier.
                if prev_ts is not None:
                    gap = (ts - prev_ts).total_seconds()
                    sleep_time = gap / self.speed_multiplier
                    if 0 < sleep_time < 5:
                        # Cap at 5s to avoid hanging on large gaps
                        # (e.g. if the OBD disconnected briefly mid-trip).
                        time.sleep(sleep_time)

                prev_ts = ts

                # Reconstruct the nested reading and feed it to the pipeline.
                reading = _row_to_reading(row)
                window_df = self._buffer.add(reading)

                if window_df is not None:
                    # A full 30-second window is ready — score it.
                    result = self._scorer.score_window(window_df, reading)
                    self.latest_result = result
                    print(
                        f"Replay window {self._scorer.windows_scored}: "
                        f"{result['alert']} (score={result['score']})"
                    )

            print("Replay complete.")

        except FileNotFoundError:
            print(f"Replay: file not found: {self.csv_path}")
        except Exception as e:
            print(f"Replay error: {e}")
            raise