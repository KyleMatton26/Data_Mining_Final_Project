import csv
import time
from collections import deque

DATA_FILE = "data/datatraining.txt"

# Choosing 30 since there is a data point at every minute so we use a sliding window for 30 mins
WINDOW_SIZE = 30
STREAM_DELAY = 0.00
PRINT_EVERY = 30

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
FEATURE_SCALES = {
    "Temperature": 25.0,
    "Humidity": 100.0,
    "Light": 1000.0,
    "CO2": 2000.0,
    "HumidityRatio": 0.01,
}


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


def normalize_features(record) -> list[float]:
    return [record[feat] / FEATURE_SCALES[feat] for feat in FEATURES]


def distance_squared(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


class StreamingKMeans:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters: int = n_clusters
        self.centroids: list[list[float]] = []
        self.counts: list[int] = []
        self.cluster_truth: list[dict[int, int]] = [
            {0: 0, 1: 0} for _ in range(n_clusters)
        ]

    def predict(self, features: list[float]) -> int:
        if not self.centroids:
            return 0
        distances = [distance_squared(features, centroid) for centroid in self.centroids]
        return min(range(len(distances)), key=lambda i: distances[i])

    def update(self, features: list[float], occupancy: int) -> int:
        if len(self.centroids) < self.n_clusters:
            self.centroids.append(features.copy())
            self.counts.append(0)
            label = len(self.centroids) - 1
        else:
            label = self.predict(features)

        self.counts[label] += 1
        self.cluster_truth[label][occupancy] += 1

        centroid = self.centroids[label]
        count = self.counts[label]
        for i in range(len(centroid)):
            centroid[i] += (features[i] - centroid[i]) / count

        return label

    def best_cluster_label(self, cluster_id: int) -> int:
        truth_counts = self.cluster_truth[cluster_id]
        return 1 if truth_counts[1] >= truth_counts[0] else 0

    def cluster_mapping(self) -> dict[int, int]:
        return {cluster_id: self.best_cluster_label(cluster_id) for cluster_id in range(len(self.centroids))}


def stream_kmeans(filepath) -> None:
    clusterer = StreamingKMeans(n_clusters=2)
    total_count = 0
    predictions: list[int] = []
    truths: list[int] = []

    print(f"Starting streaming K-means from: {filepath}")
    print(f"Learning k=2 clusters with incoming records")

    with open(filepath, newline="") as f:
        raw_header = f.readline().strip()
        fieldnames = ["_idx"]

        for h in raw_header.split(","):
            fieldnames.append(h.strip('"'))

        reader = csv.DictReader(f, fieldnames=fieldnames)

        for row in reader:
            total_count += 1
            record = parse_record(row)
            features = normalize_features(record)
            label = clusterer.update(features, record["Occupancy"])
            predictions.append(label)
            truths.append(record["Occupancy"])

            if total_count % PRINT_EVERY == 0:
                majority = clusterer.cluster_mapping()
                mapped = [majority[p] for p in predictions]
                correct = sum(1 for p, t in zip(mapped, truths) if p == t)
                print(f"Processed {total_count} records | current accuracy: {correct/total_count:.3%}")

            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)

    mapping = clusterer.cluster_mapping()
    mapped_predictions = [mapping[p] for p in predictions]
    correct_count = sum(1 for p, t in zip(mapped_predictions, truths) if p == t)
    accuracy = correct_count / total_count if total_count else 0.0

    print(f"\n{'='*60}")
    print(f"Stream complete. Total records processed: {total_count}")
    print(f"Final accuracy: {accuracy:.3%}")
    print(f"Cluster counts:")

    for cluster_id in range(len(clusterer.centroids)):
        cluster_size = clusterer.counts[cluster_id]
        truth_counts = clusterer.cluster_truth[cluster_id]
        occupied = truth_counts[1]
        unoccupied = truth_counts[0]
        print(f"  Cluster {cluster_id}: {cluster_size} records | occupied={occupied} unoccupied={unoccupied}")

    print(f"\nCluster centroids (original feature scale):")
    for cluster_id, centroid in enumerate(clusterer.centroids):
        unscaled = [centroid[i] * FEATURE_SCALES[FEATURES[i]] for i in range(len(FEATURES))]
        centroid_str = ", ".join(f"{FEATURES[i]}={unscaled[i]:.3f}" for i in range(len(FEATURES)))
        print(f"  Cluster {cluster_id}: {centroid_str}")


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
    stream_kmeans(DATA_FILE)
