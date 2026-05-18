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
from logger import flatten_reading


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
        # Flatten the nested JSON reading to a flat dict before storing,
        # so _build_window() can reference columns by name (e.g. "speed")
        # without having to unpack nested dicts.
        self._readings.append(flatten_reading(reading))
        self._since_last_window += 1

        if len(self._readings) < BUFFER_SIZE:
            return None

        if self._since_last_window >= WINDOW_SIZE:
            self._since_last_window = 0
            return self._build_window()

        return None

    def _build_window(self):
        """
        Extract the current buffer contents, compute derived signals,
        and summarize into a single-row feature DataFrame.

        This matches exactly what the notebook's extract_windows() function
        does during training — each 30-second window becomes ONE row of
        40 features (10 signals × 4 statistics each).

        The 10 signals are:
            acceleration, jerk, rpm_rate, throttle_rate,
            speed_rpm_ratio, throttle_rpm_ratio,
            accel_x, accel_y, accel_mag, gyro_z

        The 4 statistics per signal are:
            mean, std, p25, p75

        Returns:
            A single-row DataFrame with exactly 40 named feature columns.
        """
        df = pd.DataFrame(list(self._readings))

        # Parse timestamps and set as index for resampling.
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
        df = df.set_index("ts").sort_index()

        # Resample to exactly 1Hz — handles irregular arrival times.
        df = df.resample("1s").mean(numeric_only=True)
        df = df.ffill(limit=2)

        # --- Compute derived signals ---
        df["acceleration"] = df["speed"].diff()
        df["jerk"] = df["acceleration"].diff()
        df["rpm_rate"] = df["rpm"].diff()
        df["throttle_rate"] = df["throttle"].diff()
        df["speed_rpm_ratio"] = df["speed"] / (df["rpm"] + 100)
        df["throttle_rpm_ratio"] = df["throttle"] / (df["rpm"] + 100)
        df["accel_mag"] = np.sqrt(
            df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
        )

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Use only the window rows, not the lookback rows.
        window = df.tail(WINDOW_SIZE)

        # --- Compute summary statistics ---
        # These are the 10 signals the model was trained on.
        SIGNALS = [
            "acceleration", "jerk", "rpm_rate", "throttle_rate",
            "speed_rpm_ratio", "throttle_rpm_ratio",
            "accel_x", "accel_y", "accel_mag", "gyro_z"
        ]

        # Build one row of 40 features: signal_mean, signal_std,
        # signal_p25, signal_p75 for each of the 10 signals.
        features = {}
        for signal in SIGNALS:
            if signal in window.columns:
                col = window[signal]
                features[f"{signal}_mean"] = col.mean()
                features[f"{signal}_std"] = col.std()
                features[f"{signal}_p25"] = col.quantile(0.25)
                features[f"{signal}_p75"] = col.quantile(0.75)
            else:
                # Signal missing — fill with zeros so scorer doesn't crash.
                features[f"{signal}_mean"] = 0.0
                features[f"{signal}_std"] = 0.0
                features[f"{signal}_p25"] = 0.0
                features[f"{signal}_p75"] = 0.0

        # Return as a single-row DataFrame.
        # The scorer calls predict_proba on this, so shape must be (1, 40).
        return pd.DataFrame([features])