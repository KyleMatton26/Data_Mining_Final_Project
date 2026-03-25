import csv
import time
from collections import deque

DATA_FILE = "data/datatraining.txt"

# Choosing 30 since there is a data point at every minute so we use a sliding window for 30 mins
WINDOW_SIZE = 30
STREAM_DELAY = 0.00
PRINT_EVERY = 30

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def parse_record(row) -> dict[str, float | int | str]:
    return {
        "date": row["date"].strip('"'),
        "Temperature": float(row["Temperature"]),
        "Humidity": float(row["Humidity"]),
        "Light": float(row["Light"]),
        "CO2": float(row["CO2"]),
        "HumidityRatio": float(row["HumidityRatio"]),
        "Occupancy": int(row["Occupancy"]),
    }


def window_stats(window) -> dict[str, dict[str, float]]:

    stats = {}
    for feature in FEATURES:

        values = []

        for r in window:
            values.append(r[feature])

        n = len(values)
        mean = sum(values) / n

        variance = 0

        for v in values:
            variance += (v - mean) ** 2

        variance /= n

        std = variance ** 0.5

        feat_min = values[0]
        feat_max = values[0]

        for v in values:

            if v < feat_min:
                feat_min = v

            if v > feat_max:
                feat_max = v

        stats[feature] = {"mean": mean, "std": std, "min": feat_min, "max": feat_max}

    return stats


def print_summary(record_num, record, window, occupied_count, total_count) -> None:

    stats = window_stats(window)
    occ_pct = 100 * occupied_count / total_count if total_count else 0

    print(f"\n{'='*60}")
    print(f"Record #{record_num}  |  Timestamp: {record['date']}")
    print(f"Ground truth: {'OCCUPIED' if record['Occupancy'] == 1 else 'UNOCCUPIED'}")
    print(f"Occupancy so far: {occupied_count}/{total_count} ({occ_pct:.1f}%)")
    print(f"\nRolling window ({len(window)} records):")
    print(f"  {'Feature':<16} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*56}")

    for feat, s in stats.items():
        print(f"  {feat:<16} {s['mean']:>8.3f}  {s['std']:>8.3f}  {s['min']:>8.3f}  {s['max']:>8.3f}")


def stream(filepath) -> None:
    window = deque(maxlen=WINDOW_SIZE)
    total_count = 0
    occupied_count = 0

    print(f"Starting stream from: {filepath}")
    print(f"Window size: {WINDOW_SIZE} records | Print every: {PRINT_EVERY} records\n")

    with open(filepath, newline="") as f:

        raw_header = f.readline().strip()
        fieldnames = ["_idx"]

        for h in raw_header.split(","):
            fieldnames.append(h.strip('"'))

        reader = csv.DictReader(f, fieldnames=fieldnames)

        for row in reader:

            record = parse_record(row)
            window.append(record)

            total_count += 1

            if record["Occupancy"] == 1:
                occupied_count += 1

            if total_count % PRINT_EVERY == 0:
                print_summary(total_count, record, window, occupied_count, total_count)

            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)

    print(f"\n{'='*60}")
    print(f"Stream complete. Total records processed: {total_count}")
    print(f"Overall occupancy: {occupied_count}/{total_count} ({100*occupied_count/total_count:.1f}%)")

    final_stats = window_stats(window)

    print(f"\nFinal window stats ({len(window)} records):")
    print(f"  {'Feature':<16} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*56}")

    for feat, s in final_stats.items():
        print(f"  {feat:<16} {s['mean']:>8.3f}  {s['std']:>8.3f}  {s['min']:>8.3f}  {s['max']:>8.3f}")


if __name__ == "__main__":
    stream(DATA_FILE)
