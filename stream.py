import csv
import time
from collections import deque

DATA_FILE = "data/datatraining.txt"

WINDOW_SIZE = 30
STREAM_DELAY = 0.00
PRINT_EVERY = 30

STREAM_WINDOW = 500
DBSCAN_EPS = 0.01
DBSCAN_MIN_SAMPLES = 10

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

def _region_query(points: list[list[float]], idx: int, eps: float, eps_sq: float) -> list[int]:
    """Indices of all points within eps Euclidean distance of points[idx]."""
    p = points[idx]
    result = []
    for j, q in enumerate(points):
        skip = False
        for pk, qk in zip(p, q):
            if abs(pk - qk) > eps:
                skip = True
                break
        if not skip and distance_squared(p, q) <= eps_sq:
            result.append(j)
    return result


def dbscan(points: list[list[float]], eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> list[int]:
    """
    DBSCAN clustering on a list of feature vectors.
    Returns cluster labels; -1 means noise.
    """
    n = len(points)
    labels = [-1] * n
    visited = [False] * n
    eps_sq = eps * eps
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = _region_query(points, i, eps, eps_sq)
        if len(nbrs) < min_samples:
            continue  # noise for now; may become a border point later

        labels[i] = cluster_id
        seed_set = set(nbrs)
        seed_set.discard(i)

        while seed_set:
            q = seed_set.pop()
            if not visited[q]:
                visited[q] = True
                q_nbrs = _region_query(points, q, eps, eps_sq)
                if len(q_nbrs) >= min_samples:
                    seed_set.update(q_nbrs)
            if labels[q] == -1:
                labels[q] = cluster_id

        cluster_id += 1

    return labels

def compute_silhouette(features: list[list[float]], labels: list[int], max_samples: int = 500) -> float | None:
    """
    Silhouette Score, excluding noise points (label -1).
    Uses systematic sampling when valid points exceed max_samples so that the
    O(n^2) distance pass stays practical in pure Python.
    Returns None when fewer than 2 non-noise clusters exist.
    """
    valid_idx = [i for i in range(len(labels)) if labels[i] != -1]
    if len(set(labels[i] for i in valid_idx)) < 2:
        return None

    # Systematic downsample: take every k-th valid index
    if len(valid_idx) > max_samples:
        step = len(valid_idx) // max_samples
        valid_idx = valid_idx[::step][:max_samples]

    if len(set(labels[i] for i in valid_idx)) < 2:
        return None

    clusters: dict[int, list[int]] = {}
    for i in valid_idx:
        cid = labels[i]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(i)

    scores = []
    for i in valid_idx:
        cid = labels[i]
        same = clusters[cid]

        a_i = (
            sum(distance_squared(features[i], features[j]) ** 0.5 for j in same if j != i)
            / (len(same) - 1)
            if len(same) > 1
            else 0.0
        )

        b_i = float("inf")
        for other_cid, other_idx in clusters.items():
            if other_cid == cid:
                continue
            mean_d = (
                sum(distance_squared(features[i], features[j]) ** 0.5 for j in other_idx)
                / len(other_idx)
            )
            if mean_d < b_i:
                b_i = mean_d

        if b_i == float("inf"):
            continue
        denom = max(a_i, b_i)
        scores.append((b_i - a_i) / denom if denom > 0 else 0.0)

    return sum(scores) / len(scores) if scores else None


def print_eval_metrics(y_true: list[int], y_pred: list[int], features: list[list[float]] | None = None, labels: list[int] | None = None) -> None:
    """
    Print Precision, Recall, F1, Confusion Matrix, and (when features + labels are
    supplied) Silhouette Score.  features/labels refer to raw cluster assignments,
    not the occupancy-mapped predictions.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"    {'':>18} Pred_Unocc  Pred_Occ")
    print(f"    True_Unoccupied: {tn:>10}  {fp:>8}")
    print(f"    True_Occupied:   {fn:>10}  {tp:>8}")

    if features is not None and labels is not None:
        sil = compute_silhouette(features, labels)
        if sil is not None:
            print(f"  Silhouette Score:     {sil:.4f}  (higher is better, [-1, 1])")
        else:
            print(f"  Silhouette Score:     N/A  (fewer than 2 non-noise clusters)")

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


# ---------------------------------------------------------------------------
# Streaming DBSCAN
# ---------------------------------------------------------------------------

class StreamingDBSCAN:
    """
    Adapts batch DBSCAN to a streaming setting by maintaining a sliding window
    of recent points and re-fitting at each reporting checkpoint.

    To prevent cluster collapse during periods when one occupancy class is absent
    from the window, we maintain a running centroid per class and inject exactly
    min_samples identical copies of it into the DBSCAN input.  Identical points
    are within eps=0 of each other, so they always form a core point regardless
    of eps.  The anchor points are excluded from evaluation — only window labels
    are returned.
    """

    def __init__(self, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES, window_size: int = STREAM_WINDOW):
        self.eps = eps
        self.min_samples = min_samples
        self.feature_window: deque = deque(maxlen=window_size)
        self.truth_window: deque = deque(maxlen=window_size)
        self.class_centroids: dict[int, list[float] | None] = {0: None, 1: None}
        self.class_counts: dict[int, int] = {0: 0, 1: 0}

    def update(self, features: list[float], occupancy: int) -> None:
        self.feature_window.append(features)
        self.truth_window.append(occupancy)
        self.class_counts[occupancy] += 1
        n = self.class_counts[occupancy]
        if self.class_centroids[occupancy] is None:
            self.class_centroids[occupancy] = features.copy()
        else:
            c = self.class_centroids[occupancy]
            for i in range(len(c)):
                c[i] += (features[i] - c[i]) / n

    def fit(self) -> tuple[list[int], int, list[list[float]], list[int]]:
        """
        Run DBSCAN on (window + centroid anchors).
        Returns (window_labels, n_clusters_total, feat_list, true_list).
        window_labels covers only window points; n_clusters_total counts all
        clusters including those formed solely by anchors.
        """
        feat_list = list(self.feature_window)
        true_list = list(self.truth_window)
        n_window = len(feat_list)

        anchors: list[list[float]] = []
        for centroid in self.class_centroids.values():
            if centroid is not None:
                anchors.extend([centroid] * self.min_samples)

        all_labels = dbscan(feat_list + anchors, self.eps, self.min_samples)
        window_labels = all_labels[:n_window]
        n_clusters_total = len(set(all_labels)) - (1 if -1 in all_labels else 0)

        return window_labels, n_clusters_total, feat_list, true_list

    @staticmethod
    def map_clusters(labels: list[int], y_true: list[int]) -> dict[int, int]:
        """Majority-vote mapping from cluster IDs to occupancy labels. Noise is skipped."""
        mapping: dict[int, int] = {}
        for cid in set(labels):
            if cid == -1:
                continue
            occupied = sum(1 for i, lbl in enumerate(labels) if lbl == cid and y_true[i] == 1)
            total    = sum(1 for lbl in labels if lbl == cid)
            mapping[cid] = 1 if occupied >= total - occupied else 0
        return mapping

    @staticmethod
    def predict_labels(labels: list[int], features: list[list[float]], mapping: dict[int, int]) -> list[int]:
        """Map cluster labels to occupancy predictions; noise points go to nearest centroid."""
        if not mapping:
            return [0] * len(labels)

        centroids: dict[int, list[float]] = {}
        for cid in mapping:
            pts = [features[i] for i, lbl in enumerate(labels) if lbl == cid]
            d = len(pts[0])
            centroids[cid] = [sum(pt[k] for pt in pts) / len(pts) for k in range(d)]

        predicted = []
        for i, lbl in enumerate(labels):
            if lbl == -1:
                nearest = min(centroids, key=lambda c: distance_squared(features[i], centroids[c]))
                predicted.append(mapping[nearest])
            else:
                predicted.append(mapping.get(lbl, 0))
        return predicted
    
def stream_kmeans(filepath) -> None:
    clusterer = StreamingKMeans(n_clusters=2)
    total_count = 0
    feature_window: deque = deque(maxlen=STREAM_WINDOW)
    label_window: deque = deque(maxlen=STREAM_WINDOW)
    truth_window: deque = deque(maxlen=STREAM_WINDOW)

    print(f"Starting streaming K-means from: {filepath}")
    print(f"k=2, window={STREAM_WINDOW} records for periodic evaluation")

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
            feature_window.append(features)
            label_window.append(label)
            truth_window.append(record["Occupancy"])

            if total_count % PRINT_EVERY == 0:
                win_labels = list(label_window)
                win_truths = list(truth_window)
                majority = clusterer.cluster_mapping()
                mapped = [majority[p] for p in win_labels]
                correct = sum(1 for p, t in zip(mapped, win_truths) if p == t)
                win_size = len(win_labels)
                print(f"Processed {total_count} records | window={win_size} | window accuracy: {correct/win_size:.3%}")

            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)

    feat_list = list(feature_window)
    lbl_list = list(label_window)
    true_list = list(truth_window)
    win_size = len(feat_list)
    mapping = clusterer.cluster_mapping()
    mapped_predictions = [mapping[p] for p in lbl_list]
    correct_count = sum(1 for p, t in zip(mapped_predictions, true_list) if p == t)
    accuracy = correct_count / win_size if win_size else 0.0

    print(f"\n{'='*60}")
    print(f"Stream complete. Total records processed: {total_count}")
    print(f"Final window: {win_size} records | Final window accuracy: {accuracy:.3%}")
    print(f"\nCluster counts (all data):")

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

    print(f"\nValidation Metrics (last {win_size} records):")
    print_eval_metrics(true_list, mapped_predictions, features=feat_list, labels=lbl_list)


def stream_dbscan(filepath) -> None:
    clusterer = StreamingDBSCAN()
    total_count = 0

    print(f"\nStarting streaming DBSCAN from: {filepath}")
    print(f"eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}, window={STREAM_WINDOW} records")
    print(f"DBSCAN is re-fit on the sliding window at each {PRINT_EVERY}-record checkpoint.")

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
            clusterer.update(features, record["Occupancy"])

            if total_count % PRINT_EVERY == 0 and total_count >= clusterer.min_samples * 2:
                labels, n_all, feat_list, true_list = clusterer.fit()
                n_win    = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise  = labels.count(-1)
                win_size = len(feat_list)
                mapping  = StreamingDBSCAN.map_clusters(labels, true_list)
                predicted = StreamingDBSCAN.predict_labels(labels, feat_list, mapping)
                correct  = sum(1 for p, t in zip(predicted, true_list) if p == t)
                # n_win: clusters with real window points; n_all: includes anchor-only clusters
                print(
                    f"Processed {total_count} records | window={win_size} | "
                    f"clusters={n_win} ({n_all} w/memory) | "
                    f"noise={n_noise} ({100*n_noise/win_size:.1f}%) | "
                    f"window accuracy: {correct/win_size:.3%}"
                )

            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)

    labels, n_all, feat_list, true_list = clusterer.fit()
    n_win  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = labels.count(-1)
    win_size = len(feat_list)
    mapping  = StreamingDBSCAN.map_clusters(labels, true_list)
    predicted = StreamingDBSCAN.predict_labels(labels, feat_list, mapping)
    correct_count = sum(1 for p, t in zip(predicted, true_list) if p == t)
    accuracy = correct_count / win_size if win_size else 0.0

    print(f"\n{'='*60}")
    print(f"Stream complete. Total records processed: {total_count}")
    print(f"Final window: {win_size} records | clusters={n_win} ({n_all} w/memory) | noise={n_noise} ({100*n_noise/win_size:.1f}%)")
    print(f"Final window accuracy: {accuracy:.3%}")

    print(f"\nCluster breakdown (last {win_size} records):")
    for cid in sorted(set(labels)):
        indices = [i for i, lbl in enumerate(labels) if lbl == cid]
        cluster_size = len(indices)
        if cid == -1:
            print(f"  Noise:     {cluster_size} records")
        else:
            occupied  = sum(1 for i in indices if true_list[i] == 1)
            label_str = "occupied" if mapping.get(cid) == 1 else "unoccupied"
            print(
                f"  Cluster {cid}: {cluster_size} records | "
                f"occupied={occupied} unoccupied={cluster_size - occupied} "
                f"-> mapped to {label_str}"
            )

    print(f"\nValidation Metrics (last {win_size} records):")
    print_eval_metrics(true_list, predicted, features=feat_list, labels=labels)


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
    stream_dbscan(DATA_FILE)
