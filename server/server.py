"""
server.py — Flask Entry Point

Responsibility: Receive sensor readings from the Android app (live mode)
or serve scores from a replaying CSV (demo mode), and return alert levels
as JSON.

This file ties together all other modules:
  - logger.py   : writes every raw reading to CSV
  - buffer.py   : accumulates readings into 30-second windows
  - scorer.py   : scores windows and produces alert levels
  - replay.py   : replays a recorded CSV for demo day

Endpoints:
  POST /score         — live mode only. Accepts one sensor reading,
                        returns the current alert level.
  GET  /latest_score  — replay mode only. Returns the most recent
                        score produced by the replay engine.
  GET  /health        — always available. Returns 200 OK so the Android
                        app can check the server is reachable before
                        starting a trip.

Usage:
  Live mode:
    python server.py --mode live

  Replay mode:
    python server.py --mode replay --file data/logs/trip_abc123.csv

  Replay at 2x speed (useful for demos):
    python server.py --mode replay --file data/logs/trip_abc123.csv --speed 2.0
"""

import argparse

from flask import Flask, request, jsonify

from logger import TripLogger
from buffer import SensorBuffer
from scorer import TripScorer
from replay import ReplayEngine


app = Flask(__name__)

# ── Per-session state (live mode) ─────────────────────────────────────────────
# Each session_id gets its own logger, buffer, and scorer.
# Keyed by session_id string sent by the Android app.
# In practice there is only ever one active session at a time,
# but using a dict makes the code correct even if two phones connect.
_sessions = {}

# ── Replay engine (replay mode) ───────────────────────────────────────────────
# Set to a ReplayEngine instance when --mode replay is used.
# None in live mode.
_replay_engine = None

# ── Server mode ───────────────────────────────────────────────────────────────
# Set from command-line args at startup. Either "live" or "replay".
_mode = None


def _get_session(session_id):
    """
    Return the (logger, buffer, scorer) tuple for this session_id.

    Creates a new set of objects if this session_id hasn't been seen
    before. This is called on every POST /score request.

    Args:
        session_id: String identifier sent by the Android app.

    Returns:
        Tuple of (TripLogger, SensorBuffer, TripScorer).
    """
    if session_id not in _sessions:
        print(f"New session: {session_id}")
        _sessions[session_id] = (
            TripLogger(session_id),
            SensorBuffer(),
            TripScorer(),
        )
    return _sessions[session_id]


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    The Android app calls this when it first connects to confirm the
    server is reachable and to find out which mode it's running in.

    Returns:
        JSON with status and current mode.
    """
    return jsonify({"status": "ok", "mode": _mode})


@app.route("/score", methods=["POST"])
def score():
    """
    Accept one sensor reading and return the current alert level.

    Live mode only. Called by the Android app once per second.

    Expected request body: JSON matching the contract in server.py docstring.
    Returns: JSON alert dict from TripScorer.score_window(), or a
             holding response if not enough data has accumulated yet.
    """
    if _mode != "live":
        return jsonify({"error": "Server is in replay mode. Use GET /latest_score."}), 400

    # Parse the incoming JSON.
    reading = request.get_json(silent=True)
    if reading is None:
        return jsonify({"error": "Invalid JSON body."}), 400

    session_id = reading.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id."}), 400

    # Get or create the objects for this session.
    logger, buffer, scorer = _get_session(session_id)

    # Log the raw reading to CSV immediately.
    logger.log(reading)

    # Feed to the buffer. Returns a DataFrame if a full window is ready,
    # or None if we still need more data.
    window_df = buffer.add(reading)

    if window_df is not None:
        # A full 30-second window is ready — score it and return result.
        result = scorer.score_window(window_df, reading)
        return jsonify(result)

    # Not enough data yet — return a holding response so the Android
    # app has something to display during the first 30 seconds.
    return jsonify({
        "alert":          "GREEN",
        "score":          0.0,
        "s_traj":         0.0,
        "s_loc":          0.0,
        "s_cont":         0.35,
        "windows_scored": scorer.windows_scored,
        "reason":         "Collecting data...",
    })


@app.route("/latest_score", methods=["GET"])
def latest_score():
    """
    Return the most recent score from the replay engine.

    Replay mode only. The Android app polls this once per second
    while the replay is running.

    Returns:
        JSON alert dict, or a waiting response if the replay hasn't
        produced its first window yet.
    """
    if _mode != "replay":
        return jsonify({"error": "Server is in live mode. Use POST /score."}), 400

    if _replay_engine.latest_result is None:
        # Replay has started but hasn't scored a full window yet.
        return jsonify({
            "alert":          "GREEN",
            "score":          0.0,
            "s_traj":         0.0,
            "s_loc":          0.0,
            "s_cont":         0.35,
            "windows_scored": 0,
            "reason":         "Replay starting...",
        })

    return jsonify(_replay_engine.latest_result)


def _parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with attributes: mode, file, speed.
    """
    parser = argparse.ArgumentParser(description="Vehicle Theft Detection Server")
    parser.add_argument(
        "--mode",
        choices=["live", "replay"],
        required=True,
        help="live: accept readings from Android app. replay: replay a recorded CSV."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the CSV file to replay (required in replay mode)."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 1.0 = real time, 2.0 = twice as fast."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _mode = args.mode

    if _mode == "replay":
        if not args.file:
            print("Error: --file is required in replay mode.")
            exit(1)
        # Create and start the replay engine before Flask starts
        # so it begins processing immediately.
        _replay_engine = ReplayEngine(args.file, speed_multiplier=args.speed)
        _replay_engine.start()

    print(f"Starting server in {_mode.upper()} mode on http://0.0.0.0:5000")
    # host="0.0.0.0" makes the server reachable from the Android phone
    # over WiFi, not just from localhost.
    # debug=False is important — debug mode runs two processes which
    # would start the replay engine twice.
    app.run(host="0.0.0.0", port=5000, debug=False)