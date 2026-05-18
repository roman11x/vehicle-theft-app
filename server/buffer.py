"""
buffer.py — Sensor Ring Buffer and Feature Preparation

Responsibility: Accumulate raw 1Hz readings from the Android app and,
every 30 seconds, produce a clean feature-ready DataFrame that the
scorer can process immediately.

Why this is needed:
  - The scoring model expects 30-second non-overlapping windows at 1Hz.
  - Real readings arrive at slightly irregular intervals (network jitter).
  - Derived signals (acceleration, jerk, rpm_rate, etc.) need the previous
    row to compute — so we keep 2 extra rows of lookback beyond the 30
    that form the window.

The main class is SensorBuffer. Call add(reading) once per incoming
reading. It returns a DataFrame when a full window is ready, or None
if more data is still needed.
"""

import pandas as pd
import numpy as np
from collections import deque


# How many seconds per window — must match what the model was trained on.
WINDOW_SIZE = 30

# Extra rows kept before the window for derivative computation.
# accel = diff(speed), jerk = diff(accel), so we need 2 prior rows.
LOOKBACK = 2

# Total rows we keep in the buffer at all times.
BUFFER_SIZE = WINDOW_SIZE + LOOKBACK


class SensorBuffer:
    """
    Accumulates incoming readings and emits 30-second windows.

    One SensorBuffer instance exists per active trip session.
    It is created fresh when a new session_id is seen and discarded
    when the trip ends.

    Usage:
        buf = SensorBuffer()
        result = buf.add(reading)   # returns DataFrame or None
        if result is not None:
            # pass result to scorer
    """

    def __init__(self):
        """
        Initialize an empty buffer.

        deque with maxlen automatically discards the oldest entry when
        a new one is added and the buffer is full — that's the "ring"
        part of ring buffer.
        """
        # Stores raw reading dicts, newest at the right.
        self._readings = deque(maxlen=BUFFER_SIZE)

        # Counts how many readings have arrived since the last window
        # was emitted. When this hits 30 we emit a new window.
        self._since_last_window = 0

    def add(self, reading: dict):
        """
        Add one incoming reading to the buffer.

        Args:
            reading: The flattened reading dict (same format as a CSV row,
                     as produced by logger.flatten_reading).

        Returns:
            A pandas DataFrame with exactly WINDOW_SIZE rows and all
            derived signal columns added, ready for the scorer.
            Returns None if not enough data has accumulated yet.
        """
        self._readings.append(reading)
        self._since_last_window += 1

        # Not enough data yet for even one window.
        if len(self._readings) < BUFFER_SIZE:
            return None

        # A full 30-second window has accumulated since the last one.
        if self._since_last_window >= WINDOW_SIZE:
            self._since_last_window = 0
            return self._build_window()

        return None

    def _build_window(self):
        """
        Extract the current buffer contents and prepare a feature DataFrame.

        Steps:
          1. Convert the deque to a DataFrame.
          2. Resample to a clean 1Hz grid (handles irregular arrival times).
          3. Compute derived signals that the model needs.
          4. Return only the last WINDOW_SIZE rows (drop the lookback rows).

        Returns:
            DataFrame with WINDOW_SIZE rows and all feature columns present.
        """
        df = pd.DataFrame(list(self._readings))

        # Parse timestamps and set as index for resampling.
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()

        # Resample to exactly 1Hz by taking the mean within each second.
        # This smooths out the slight irregularity in arrival times.
        # numeric_only=True skips the session_id string column.
        df = df.resample("1s").mean(numeric_only=True)

        # Forward-fill any gaps left by resampling (e.g. a dropped reading).
        # limit=2 means we won't fill more than 2 consecutive missing seconds.
        df = df.ffill(limit=2)

        # --- Derived signals ---
        # These are the signals the model was trained on that are not
        # directly measured but computed from the raw OBD/IMU signals.

        # acceleration: how fast speed is changing (m/s² proxy from km/h)
        df["acceleration"] = df["speed"].diff()

        # jerk: how fast acceleration is changing (smoothness of driving)
        df["jerk"] = df["acceleration"].diff()

        # rpm_rate: how fast the engine RPM is changing
        df["rpm_rate"] = df["rpm"].diff()

        # throttle_rate: how quickly the driver is pressing/releasing the pedal
        df["throttle_rate"] = df["throttle"].diff()

        # speed_rpm_ratio: proxy for which gear the driver is in
        # Adding 100 to rpm avoids division by zero at engine idle/off
        df["speed_rpm_ratio"] = df["speed"] / (df["rpm"] + 100)

        # throttle_rpm_ratio: how the throttle pedal maps to engine response
        df["throttle_rpm_ratio"] = df["throttle"] / (df["rpm"] + 100)

        # accel_mag: total acceleration magnitude, rotation-invariant.
        # This is the one IMU feature that works even if the phone mount
        # angle is slightly off, because it doesn't depend on axis alignment.
        df["accel_mag"] = np.sqrt(
            df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2
        )

        # Replace any inf or NaN with 0.
        # NaN appears on the first row of diff() (no previous row to subtract).
        # inf can appear if rpm+100 somehow rounds to 0 (shouldn't happen but safe).
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Return only the window rows, not the lookback rows.
        # The lookback rows were only needed to give diff() a previous value
        # on the first row of the window — we don't score them.
        return df.tail(WINDOW_SIZE).reset_index(drop=True)