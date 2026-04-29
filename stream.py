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
    """Parse a raw CSV row into a typed record dictionary.

    Args:
        row: A csv.DictReader row containing sensor and occupancy fields.

    Returns:
        A dict with keys: ``date`` (str), ``Temperature``, ``Humidity``,
        ``Light``, ``CO2``, ``HumidityRatio`` (float), and ``Occupancy`` (int).
    """
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
    """Compute descriptive statistics for each feature over a sliding window.

    Args:
        window: An iterable of record dicts, each containing all keys in
            ``FEATURES``.

    Returns:
        A dict mapping each feature name to a nested dict with keys
        ``mean``, ``std``, ``min``, and ``max``.
    """
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
    """Print a formatted snapshot of the current stream position.

    Displays the record number, timestamp, ground-truth occupancy, running
    occupancy percentage, and a per-feature statistics table for the current
    sliding window.

    Args:
        record_num: The 1-based index of the current record in the stream.
        record: The current parsed record dict (output of ``parse_record``).
        window: The current sliding window of record dicts.
        occupied_count: Number of occupied records seen so far.
        total_count: Total records processed so far.
    """
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
    """Return a normalized feature vector for a single record.

    Each feature is divided by its corresponding scale factor in
    ``FEATURE_SCALES`` so that all dimensions lie roughly in [0, 1].

    Args:
        record: A parsed record dict (output of ``parse_record``).

    Returns:
        A list of floats, one per feature in ``FEATURES``, in the same order.
    """
    return [record[feat] / FEATURE_SCALES[feat] for feat in FEATURES]


def distance_squared(a: list[float], b: list[float]) -> float:
    """Compute the squared Euclidean distance between two feature vectors.

    Args:
        a: First feature vector.
        b: Second feature vector. Must have the same length as ``a``.

    Returns:
        The sum of squared element-wise differences.
    """
    return sum((x - y) ** 2 for x, y in zip(a, b))

def _region_query(points: list[list[float]], idx: int, eps: float, eps_sq: float) -> list[int]:
    """Return indices of all points within ``eps`` Euclidean distance of a query point.

    Uses a per-dimension absolute-difference pre-filter before computing the
    full squared distance to skip obviously distant points cheaply.

    Args:
        points: List of feature vectors.
        idx: Index of the query point within ``points``.
        eps: Neighborhood radius (Euclidean).
        eps_sq: Pre-computed ``eps ** 2`` to avoid repeated squaring.

    Returns:
        A list of indices (including ``idx`` itself) whose Euclidean distance
        to ``points[idx]`` is at most ``eps``.
    """
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
    """Run DBSCAN clustering on a list of feature vectors.

    Args:
        points: List of feature vectors to cluster.
        eps: Maximum Euclidean distance between two points for them to be
            considered neighbors. Defaults to ``DBSCAN_EPS``.
        min_samples: Minimum number of neighbors (including the point itself)
            required for a point to be a core point. Defaults to
            ``DBSCAN_MIN_SAMPLES``.

    Returns:
        A list of integer cluster labels, one per input point. A label of
        ``-1`` indicates that the point was classified as noise.
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
    """Online K-means clusterer that updates centroids incrementally.

    Each call to ``update`` assigns the incoming point to the nearest
    centroid and adjusts that centroid via a running mean, requiring only
    O(k) memory regardless of stream length.
    """

    def __init__(self, n_clusters: int = 2):
        """Initialize the clusterer with empty centroids.

        Args:
            n_clusters: Number of clusters to maintain. Defaults to 2.
        """
        self.n_clusters: int = n_clusters
        self.centroids: list[list[float]] = []
        self.counts: list[int] = []
        self.cluster_truth: list[dict[int, int]] = [
            {0: 0, 1: 0} for _ in range(n_clusters)
        ]

    def predict(self, features: list[float]) -> int:
        """Assign a feature vector to the nearest cluster centroid.

        Args:
            features: Normalized feature vector to classify.

        Returns:
            Index of the nearest centroid, or ``0`` if no centroids exist yet.
        """
        if not self.centroids:
            return 0
        distances = [distance_squared(features, centroid) for centroid in self.centroids]
        return min(range(len(distances)), key=lambda i: distances[i])

    def update(self, features: list[float], occupancy: int) -> int:
        """Assign a point to a cluster and update the centroid incrementally.

        If fewer than ``n_clusters`` centroids have been initialized, the
        point seeds a new centroid. Otherwise the point is assigned to the
        nearest existing centroid, whose running mean is then updated.

        Args:
            features: Normalized feature vector of the incoming data point.
            occupancy: Ground-truth occupancy label (0 or 1) used to track
                per-cluster label distributions for later majority voting.

        Returns:
            The cluster index to which the point was assigned.
        """
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
        """Return the majority occupancy label for a cluster via ground-truth counts.

        Args:
            cluster_id: Index of the cluster to evaluate.

        Returns:
            ``1`` if more occupied than unoccupied points were assigned to the
            cluster, otherwise ``0``.
        """
        truth_counts = self.cluster_truth[cluster_id]
        return 1 if truth_counts[1] >= truth_counts[0] else 0

    def cluster_mapping(self) -> dict[int, int]:
        """Build a majority-vote mapping from cluster index to occupancy label.

        Returns:
            A dict where each key is a cluster index and the value is the
            predicted occupancy label (0 or 1) based on the accumulated
            ground-truth counts for that cluster.
        """
        return {cluster_id: self.best_cluster_label(cluster_id) for cluster_id in range(len(self.centroids))}

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
        """Ingest a new data point into the sliding window and update class centroids.

        The point is appended to both the feature and truth windows (oldest
        entries are automatically evicted when the window is full). The
        running centroid for the point's occupancy class is also updated so
        that anchor injection in ``fit`` reflects current class statistics.

        Args:
            features: Normalized feature vector of the incoming data point.
            occupancy: Ground-truth occupancy label (0 or 1).
        """
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
    """Run online K-means over a CSV data stream and print evaluation metrics.

    Reads the file one record at a time, updating a ``StreamingKMeans``
    clusterer incrementally. Every ``PRINT_EVERY`` records the window
    accuracy is printed to stdout. A final summary with per-cluster
    statistics and validation metrics is printed at the end of the stream.

    Args:
        filepath: Path to the CSV data file (must follow the training-data
            format with a quoted header row).
    """
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
    """Run sliding-window DBSCAN over a CSV data stream and print evaluation metrics.

    Reads the file one record at a time, updating a ``StreamingDBSCAN``
    clusterer. Every ``PRINT_EVERY`` records (once the window is large enough)
    DBSCAN is re-fit on the current window and a progress line is printed.
    A final summary with per-cluster breakdowns and validation metrics is
    printed at the end of the stream.

    Args:
        filepath: Path to the CSV data file (must follow the training-data
            format with a quoted header row).
    """
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
    """Stream raw sensor records from a CSV file and print rolling statistics.

    Reads the file one record at a time, maintaining a sliding window of the
    most recent ``WINDOW_SIZE`` records. Every ``PRINT_EVERY`` records a
    summary table of per-feature statistics is printed. A final summary with
    overall occupancy rates and window stats is printed after the stream ends.

    Args:
        filepath: Path to the CSV data file (must follow the training-data
            format with a quoted header row).
    """
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
