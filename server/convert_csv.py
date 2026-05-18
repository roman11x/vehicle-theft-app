"""
convert_csv.py — Raw OBD App CSV to Server Format Converter

Responsibility: Convert CSV files recorded directly by the OBD MX+
Bluetooth app into the flat format that logger.py writes and
replay.py reads.

Why this is needed:
  The OBD app saves CSVs with its own column names and a datetime
  timestamp format. The server's logger.py writes CSVs with different
  column names and ISO 8601 timestamps. Replay mode reads the server
  format, so raw OBD CSVs must be converted first.

  Raw OBD app columns (relevant ones):
    Time, Vehicle speed (km/h), Engine RPM (RPM),
    Absolute throttle position (%), Calculated load value (%),
    Mass air flow rate (g/s), Accel X (m/s²), Accel Y (m/s²),
    Accel Z (m/s²), Rotation Rate X (deg/s), Rotation Rate Y (deg/s),
    Rotation Rate Z (deg/s), Latitude (deg), Longitude (deg),
    GPS Speed (km/h), Horz Accuracy (m)

  Server format columns:
    ts, session_id, speed, rpm, throttle, engine_load, maf,
    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
    lat, lon, speed_gps, accuracy

Usage:
  python convert_csv.py --input path/to/raw.csv --output path/to/converted.csv --driver G

  The --driver flag sets the session_id in the output file so replay
  mode knows which driver this trip belongs to.
"""

import argparse
import csv
import os
from datetime import datetime

from logger import COLUMNS


# Mapping from raw OBD app column names to server column names.
# Columns not in this map are dropped.
COLUMN_MAP = {
    "Vehicle speed (km/h)":          "speed",
    "Engine RPM (RPM)":              "rpm",
    "Absolute throttle position (%)": "throttle",
    "Calculated load value (%)":     "engine_load",
    "Mass air flow rate (g/s)":      "maf",
    "Accel X (m/s²)":               "accel_x",
    "Accel Y (m/s²)":               "accel_y",
    "Accel Z (m/s²)":               "accel_z",
    "Rotation Rate X (deg/s)":       "gyro_x",
    "Rotation Rate Y (deg/s)":       "gyro_y",
    "Rotation Rate Z (deg/s)":       "gyro_z",
    "Latitude (deg)":                "lat",
    "Longitude (deg)":               "lon",
    "GPS Speed (km/h)":              "speed_gps",
    "Horz Accuracy (m)":             "accuracy",
}


def parse_obd_timestamp(raw_ts):
    """
    Convert OBD app timestamp to ISO 8601 format.

    OBD app format: '01/23/2026 09:07:49.1676 pm'
    ISO 8601 output: '2026-01-23T21:07:49.167600'

    Args:
        raw_ts: Timestamp string from the raw OBD CSV.

    Returns:
        ISO 8601 string, or empty string if parsing fails.
    """
    try:
        # Strip leading/trailing whitespace first.
        raw_ts = raw_ts.strip()
        dt = datetime.strptime(raw_ts, "%m/%d/%Y %I:%M:%S.%f %p")
        return dt.isoformat()
    except ValueError:
        return ""


def convert(input_path, output_path, driver_id):
    """
    Convert one raw OBD CSV to server format.

    Reads every row from the input file, remaps column names,
    converts the timestamp, and writes to the output file.
    Rows with unparseable timestamps are skipped with a warning.

    Args:
        input_path: Path to the raw OBD app CSV.
        output_path: Path to write the converted server-format CSV.
        driver_id: String used as session_id in the output
                   (e.g. "Y", "G", "test_drive_1").
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows_read    = 0
    rows_written = 0
    rows_skipped = 0

    with open(input_path,  newline="", encoding="utf-8-sig") as fin, \
         open(output_path, newline="", mode="w") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=COLUMNS)
        writer.writeheader()

        # Strip leading/trailing whitespace from all column names.
        # The OBD app adds a leading space to most column names.
        raw_columns = [c.strip() for c in reader.fieldnames]

        for raw_row in reader:
            rows_read += 1

            # Re-key the row with stripped column names.
            row = {k.strip(): v for k, v in raw_row.items()}

            # Convert timestamp.
            ts = parse_obd_timestamp(row.get("Time", ""))
            if not ts:
                rows_skipped += 1
                continue

            # Build output row.
            out_row = {col: None for col in COLUMNS}
            out_row["ts"]         = ts
            out_row["session_id"] = driver_id

            # Map each raw column to its server name.
            for raw_col, server_col in COLUMN_MAP.items():
                val = row.get(raw_col, "").strip()
                out_row[server_col] = val if val != "" else None

            writer.writerow(out_row)
            rows_written += 1

    print(f"Conversion complete:")
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    print(f"  Read:    {rows_read} rows")
    print(f"  Written: {rows_written} rows")
    if rows_skipped:
        print(f"  Skipped: {rows_skipped} rows (bad timestamps)")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw OBD app CSV to server replay format."
    )
    parser.add_argument("--input",  required=True, help="Path to raw OBD CSV.")
    parser.add_argument("--output", required=True, help="Path for converted output CSV.")
    parser.add_argument("--driver", required=True, help="Driver ID used as session_id.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(args.input, args.output, args.driver)