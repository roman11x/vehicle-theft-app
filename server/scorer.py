"""
scorer.py — Driving Behavior Scorer and Fusion Engine

Responsibility: Take a 30-row feature DataFrame from buffer.py,
run it through the trained XGBoost models, and produce a fused
alert level (GREEN / YELLOW / ORANGE / RED) with per-layer scores.

How scoring works:
  - Strategy A (clf_A): XGBoost trained on GPS-filtered passive data
    from road segments shared with the controlled route. 40 features.
    Best when the current road has been seen before in training data.

  - Strategy C (clf_C): XGBoost trained on all passive data using
    route-invariant features (speed-bin conditioning, maneuver
    extraction, stop-and-go profiling). 130 features.
    Used as a fallback for unfamiliar roads.

  A window is flagged as suspicious if EITHER strategy's pmax score
  falls below its threshold. Three consecutive flagged windows fire
  the unfamiliar-driver alert — this voting mechanism prevents a
  single noisy window from triggering a false alarm.

  S_traj is then fused with S_loc (GPS integrity) and S_cont
  (contextual risk) using fixed weights to produce the final score I.

  Fusion weights (renormalized with Layer 4 absent):
    I = 0.40 * S_traj + 0.33 * S_loc + 0.27 * S_cont

Alert thresholds:
    GREEN  : I < 0.30
    YELLOW : 0.30 <= I < 0.50
    ORANGE : 0.50 <= I < 0.70
    RED    : I >= 0.70
"""

import os
import pickle
import joblib
import numpy as np


# ── Paths ────────────────────────────────────────────────────────────────────
# All model files live in server/models/
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ── Fusion weights (Layer 4 absent, renormalized to sum to 1.0) ──────────────
W_TRAJ = 0.40
W_LOC  = 0.33
W_CONT = 0.27

# ── Alert thresholds ─────────────────────────────────────────────────────────
THRESH_YELLOW = 0.30
THRESH_ORANGE = 0.50
THRESH_RED    = 0.70

# ── Voting: how many consecutive suspicious windows before alerting ───────────
CONSECUTIVE_REQUIRED = 3

# ── S_cont placeholder until Yotam builds the real context module ────────────
# The real module will compute this from time-of-day, home radius,
# BLE/WiFi fingerprint, paired device check, and GPS heatmap.
S_CONT_MOCK = 0.35


def _load_models():
    """
    Load all pkl files from the models directory at startup.

    Called once when the module is first imported. Returns a dict
    containing all the objects the scorer needs at runtime.

    Raises FileNotFoundError if any required model file is missing.
    """
    def load(filename):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model file not found: {path}\n"
                f"Run the notebook and copy pkl files to server/models/"
            )
        return joblib.load(path) if filename.endswith(".pkl") and "feature" not in filename \
               else pickle.load(open(path, "rb"))

    pipeline  = pickle.load(open(os.path.join(MODELS_DIR, "pipeline_outputs.pkl"), "rb"))
    feat_cols = pickle.load(open(os.path.join(MODELS_DIR, "feature_cols.pkl"),     "rb"))

    return {
        "clf_A":           joblib.load(os.path.join(MODELS_DIR, "clf_A.pkl")),
        "clf_C":           joblib.load(os.path.join(MODELS_DIR, "clf_C.pkl")),
        "threshold_A":     pipeline["threshold_A"],
        "threshold_C":     pipeline["threshold_C"],
        "FEATURE_COLS":    feat_cols["FEATURE_COLS"],
        "ri_features_used": feat_cols["ri_features_used"],
        "FAMILY":          pipeline["FAMILY"],
    }


# Load models once at import time so the first POST /score isn't slow.
print("Loading models...")
_MODELS = _load_models()
print(f"  clf_A loaded  (threshold={_MODELS['threshold_A']:.4f})")
print(f"  clf_C loaded  (threshold={_MODELS['threshold_C']:.4f})")
print(f"  Family drivers: {_MODELS['FAMILY']}")
print("Models ready.")


def _score_window_traj(window_df):
    """
    Score one 30-row window for driving behavior anomaly (S_traj).

    Runs both Strategy A and Strategy C on the window and returns
    whether the window is suspicious plus the raw pmax values.

    A window is suspicious if EITHER strategy's pmax is below its
    threshold — we take the more sensitive reading of the two.

    Args:
        window_df: DataFrame with 30 rows from buffer.py.
                   Must contain all FEATURE_COLS and ri_features_used columns.

    Returns:
        Tuple of (is_suspicious: bool, pmax_A: float, pmax_C: float)
    """
    clf_A    = _MODELS["clf_A"]
    clf_C    = _MODELS["clf_C"]
    feat_A   = _MODELS["FEATURE_COLS"]
    feat_C   = _MODELS["ri_features_used"]
    thr_A    = _MODELS["threshold_A"]
    thr_C    = _MODELS["threshold_C"]

    # Strategy A — 40 GPS-filtered features
    # For any feature column missing from the window, fill with 0.
    X_A = np.zeros((len(window_df), len(feat_A)))
    for i, col in enumerate(feat_A):
        if col in window_df.columns:
            X_A[:, i] = window_df[col].values

    # predict_proba returns shape (n_rows, n_drivers).
    # .max(axis=1) gives the highest confidence for any known driver per row.
    # We then take the mean across the 30 rows as the window-level pmax.
    pmax_A = float(clf_A.predict_proba(X_A).max(axis=1).mean())

    # Strategy C — 130 route-invariant features
    X_C = np.zeros((len(window_df), len(feat_C)))
    for i, col in enumerate(feat_C):
        if col in window_df.columns:
            X_C[:, i] = window_df[col].values

    pmax_C = float(clf_C.predict_proba(X_C).max(axis=1).mean())

    # A window is suspicious if the driver doesn't look like
    # any known family member — i.e. pmax is LOW (below threshold).
    suspicious_A = pmax_A < thr_A
    suspicious_C = pmax_C < thr_C

    # Flag if either strategy is suspicious.
    is_suspicious = suspicious_A or suspicious_C

    return is_suspicious, pmax_A, pmax_C


def _score_gps(gps_reading):
    """
    Compute S_loc — GPS integrity score for one reading.

    Rule-based logic from the GPS spoofing/jamming notebook.
    Checks two things:
      1. Speed mismatch: if GPS speed and OBD speed differ by more
         than 20 km/h, the GPS data may be spoofed.
      2. Signal loss while moving: if GPS speed is 0 but OBD speed
         is non-zero, a jammer may be blocking the GPS signal.

    Returns a float 0.0 (clean) to 1.0 (compromised).

    Args:
        gps_reading: dict with keys 'speed_gps' and 'speed_obd'.
                     Either can be None if data is unavailable.
    """
    speed_gps = gps_reading.get("speed_gps")
    speed_obd = gps_reading.get("speed_obd")

    # Can't score without both signals — return clean.
    if speed_gps is None or speed_obd is None:
        return 0.0

    # Spoof detection: physics mismatch between GPS and OBD speed.
    if abs(speed_gps - speed_obd) > 20:
        return 1.0

    # Jamming detection: OBD says moving but GPS says stopped.
    if speed_obd > 5 and speed_gps == 0:
        return 0.8

    return 0.0


def _fuse(s_traj, s_loc, s_cont):
    """
    Combine the three layer scores into a single fused risk score I.

    Formula: I = 0.40 * S_traj + 0.33 * S_loc + 0.27 * S_cont

    Args:
        s_traj: Behavioral anomaly score (0.0–1.0).
        s_loc:  GPS integrity score (0.0–1.0).
        s_cont: Contextual risk score (0.0–1.0).

    Returns:
        Fused score I clamped to [0.0, 1.0].
    """
    I = W_TRAJ * s_traj + W_LOC * s_loc + W_CONT * s_cont
    return float(np.clip(I, 0.0, 1.0))


def _alert_level(score):
    """
    Map a fused score to a human-readable alert level.

    Args:
        score: Fused score I in [0.0, 1.0].

    Returns:
        One of "GREEN", "YELLOW", "ORANGE", "RED".
    """
    if score >= THRESH_RED:
        return "RED"
    if score >= THRESH_ORANGE:
        return "ORANGE"
    if score >= THRESH_YELLOW:
        return "YELLOW"
    return "GREEN"


class TripScorer:
    """
    Stateful scorer for one active trip session.

    One TripScorer instance is created per session (same as TripLogger).
    It maintains the consecutive-suspicious-window counter and the
    rolling S_traj across the trip.

    Usage:
        scorer = TripScorer()
        result = scorer.score_window(window_df, latest_reading)
        # result is a dict ready to be returned as JSON to the Android app.
    """

    def __init__(self):
        """Initialize counters for a fresh trip."""
        # How many windows have been scored this trip.
        self.windows_scored = 0

        # How many consecutive windows were flagged as suspicious.
        self._consecutive_suspicious = 0

        # Rolling S_traj: mean suspicion across all windows so far.
        # Suspicion per window = 1 - pmax (low pmax = high suspicion).
        self._suspicion_history = []

    def score_window(self, window_df, latest_reading):
        """
        Score one 30-row window and return a result dict for the Android app.

        Args:
            window_df:      DataFrame from buffer.py (30 rows, all feature cols).
            latest_reading: The most recent raw JSON reading, used for S_loc
                            (we need the current GPS and OBD speed for the
                            physics consistency check).

        Returns:
            Dict with keys: alert, score, s_traj, s_loc, s_cont,
                            windows_scored, reason.
        """
        self.windows_scored += 1

        # ── Layer 1: Behavioral anomaly (S_traj) ─────────────────────────────
        is_suspicious, pmax_A, pmax_C = _score_window_traj(window_df)

        # Suspicion for this window: low pmax = high suspicion.
        # We take the min pmax (most suspicious reading of the two strategies).
        min_pmax = min(pmax_A, pmax_C)
        window_suspicion = 1.0 - min_pmax
        self._suspicion_history.append(window_suspicion)

        # Rolling S_traj: 95th percentile of suspicion history.
        # Matches the methodology from the architecture doc.
        s_traj = float(np.percentile(self._suspicion_history, 95))

        # Update consecutive suspicious window counter.
        if is_suspicious:
            self._consecutive_suspicious += 1
        else:
            self._consecutive_suspicious = 0

        # ── Layer 2: GPS integrity (S_loc) ────────────────────────────────────
        gps_check = {
            "speed_gps": latest_reading.get("gps", {}).get("speed_gps"),
            "speed_obd": latest_reading.get("obd", {}).get("speed"),
        }
        s_loc = _score_gps(gps_check)

        # ── Layer 3: Context risk (S_cont) ────────────────────────────────────
        # Mocked at 0.35 until Yotam builds the real module.
        s_cont = S_CONT_MOCK

        # ── Fusion ────────────────────────────────────────────────────────────
        score = _fuse(s_traj, s_loc, s_cont)
        alert = _alert_level(score)

        # ── Reason string ─────────────────────────────────────────────────────
        # Plain English explanation for the Android app to display.
        if self._consecutive_suspicious >= CONSECUTIVE_REQUIRED:
            reason = f"Driving pattern unlike any registered driver ({self._consecutive_suspicious} consecutive windows)"
        elif s_loc > 0:
            reason = "GPS signal inconsistency detected"
        elif self.windows_scored < 10:
            reason = f"Collecting data ({self.windows_scored * 30}s of {10 * 30}s needed)"
        else:
            reason = "Driving pattern consistent with registered drivers"

        return {
            "alert":          alert,
            "score":          round(score, 3),
            "s_traj":         round(s_traj, 3),
            "s_loc":          round(s_loc, 3),
            "s_cont":         round(s_cont, 3),
            "windows_scored": self.windows_scored,
            "reason":         reason,
        }