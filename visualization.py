"""
Comments written with AI

visualization.py
----------------
Animated sliding-window cluster visualization using CO2 vs Light.

Pipeline:
  1. Load all records from the dataset (CO2, Light, Occupancy label, timestamp)
  2. Simulate the stream one record at a time, pre-computing cluster snapshots
     every RECOMPUTE_EVERY records for both KMeans and DBSCAN
  3. Animate through the snapshots at 60fps and save to SAVE_AS

Requirements:
    pip install matplotlib numpy
    brew install ffmpeg   (for MP4; skip if using GIF)
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

from stream import (
    parse_record,
    StreamingKMeans,
    StreamingDBSCAN,
    dbscan,
    compute_silhouette,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE        = "data/datatraining.txt"
STREAM_WINDOW    = 500  # number of most-recent records shown at once
RECOMPUTE_EVERY  = 30   # save a snapshot (and re-run DBSCAN) every N records
DATA_HOLD_FRAMES = 6    # how many 60fps frames each snapshot is displayed for
SAVE_AS          = "cluster_animation.mp4"  # change to .gif if no ffmpeg

# Divisors used to normalize CO2 and Light into [0, 1] for clustering
CO2_SCALE   = 2000.0
LIGHT_SCALE = 1000.0

# Color palette — dark theme
COLOR_OCCUPIED   = "#F72585"  # pink  — predicted occupied
COLOR_UNOCCUPIED = "#4361EE"  # blue  — predicted unoccupied
COLOR_NOISE      = "#666666"  # grey  — DBSCAN noise points (label == -1)
COLOR_HISTORY    = "#2A2A3A"  # dark  — records outside the active window
COLOR_BG         = "#0F0F1A"  # figure background
COLOR_PANEL      = "#16162A"  # axes background
COLOR_TEXT       = "#E8E8F0"  # labels and tick text
COLOR_GRID       = "#2A2A45"  # gridlines and spine edges


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load the dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    """
    Read the occupancy dataset and return the two features used for plotting.

    Only CO2 and Light are extracted here — they are the most visually
    separable features between occupied and unoccupied states, making them
    ideal for a 2D scatter animation.

    Returns:
        co2   (np.ndarray): raw CO2 readings in ppm
        light (np.ndarray): raw Light readings in lux
        y_true (np.ndarray[int]): ground-truth occupancy labels (0 or 1)
        dates  (list[str]): timestamp string for each record
    """
    co2_list, light_list, truth_list, date_list = [], [], [], []

    with open(filepath, newline="") as f:
        # The file has a quoted header row; strip quotes and prepend a dummy
        # index column to match the format expected by parse_record
        raw_header = f.readline().strip()
        fieldnames = ["_idx"] + [h.strip('"') for h in raw_header.split(",")]
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            record = parse_record(row)
            co2_list.append(record["CO2"])
            light_list.append(record["Light"])
            truth_list.append(record["Occupancy"])
            date_list.append(record["date"])

    return (
        np.array(co2_list),
        np.array(light_list),
        np.array(truth_list, dtype=int),
        date_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Simulate the stream and save snapshots
# ─────────────────────────────────────────────────────────────────────────────

def precompute_checkpoints(co2, light, y_true, dates):
    """
    Simulate the data stream and pre-compute cluster snapshots for animation.

    Walks through every record in order (mimicking a live stream). At each
    RECOMPUTE_EVERY-record checkpoint it:
      - Runs KMeans predict over the current sliding window
      - Re-fits DBSCAN on the current sliding window from scratch
      - Records per-point colors, window accuracy, and silhouette score

    Snapshots are stored as parallel lists (one entry per checkpoint) so the
    animator can access any snapshot in O(1) by index.

    Args:
        co2    (np.ndarray): raw CO2 values for all records
        light  (np.ndarray): raw Light values for all records
        y_true (np.ndarray): ground-truth occupancy labels
        dates  (list[str]):  timestamps, one per record

    Returns:
        Tuple of 9 parallel lists, one entry per snapshot:
            window_starts, window_ends, timestamps,
            kmeans_colors, kmeans_accuracy, kmeans_silhouette,
            dbscan_colors, dbscan_accuracy, dbscan_silhouette
    """

    # Normalize features into [0, 1] so both algorithms use the same scale
    norm_co2   = co2   / CO2_SCALE
    norm_light = light / LIGHT_SCALE
    n = len(co2)

    # Parallel snapshot lists — each index corresponds to one checkpoint
    window_starts     = []
    window_ends       = []
    timestamps        = []
    kmeans_colors     = []
    kmeans_accuracy   = []
    kmeans_silhouette = []
    dbscan_colors     = []
    dbscan_accuracy   = []
    dbscan_silhouette = []

    # KMeans is online — it updates incrementally with every record
    km = StreamingKMeans(n_clusters=2)

    print(f"Simulating stream over {n} records, snapshot every {RECOMPUTE_EVERY}...")

    for i in range(n):

        # Online KMeans update — one record at a time, no window needed
        km.update([norm_co2[i], norm_light[i]], y_true[i])

        # Skip non-checkpoint records; also wait until we have enough points
        # for DBSCAN to form at least one core point on each side
        has_enough_for_dbscan = (i + 1) >= DBSCAN_MIN_SAMPLES * 2
        is_checkpoint  = (i + 1) % RECOMPUTE_EVERY == 0 and has_enough_for_dbscan
        is_last_record = i == n - 1
        if not is_checkpoint and not is_last_record:
            continue

        # Slice the sliding window — smaller than STREAM_WINDOW early in the stream
        window_start = max(0, i - STREAM_WINDOW + 1)
        window_end   = i
        window_size  = window_end - window_start + 1

        window_co2   = norm_co2[window_start : window_end + 1]
        window_light = norm_light[window_start : window_end + 1]
        window_truth = y_true[window_start : window_end + 1]

        # Build [co2, light] feature vectors for every point in the window
        window_features = []
        for j in range(window_size):
            window_features.append([window_co2[j], window_light[j]])

        # ── K-Means ───────────────────────────────────────────────────────────
        # Map raw cluster IDs (0/1) to occupancy labels via majority vote
        kmeans_mapping = km.cluster_mapping()

        # Predict the cluster for every window point using the current centroids
        kmeans_labels = []
        for feat in window_features:
            kmeans_labels.append(km.predict(feat))

        # Convert cluster IDs to occupancy predictions
        kmeans_predictions = []
        for lbl in kmeans_labels:
            kmeans_predictions.append(kmeans_mapping[lbl])

        # Assign a display color to each point based on its prediction
        window_kmeans_colors = []
        for pred in kmeans_predictions:
            if pred == 1:
                window_kmeans_colors.append(COLOR_OCCUPIED)
            else:
                window_kmeans_colors.append(COLOR_UNOCCUPIED)

        # ── DBSCAN ────────────────────────────────────────────────────────────
        # DBSCAN has no incremental update — re-fit the entire window each time
        dbscan_labels      = dbscan(window_features, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
        dbscan_mapping     = StreamingDBSCAN.map_clusters(dbscan_labels, window_truth.tolist())
        dbscan_predictions = StreamingDBSCAN.predict_labels(dbscan_labels, window_features, dbscan_mapping)

        # Noise points (label == -1) get their own distinct color
        window_dbscan_colors = []
        for j in range(window_size):
            if dbscan_labels[j] == -1:
                window_dbscan_colors.append(COLOR_NOISE)
            elif dbscan_predictions[j] == 1:
                window_dbscan_colors.append(COLOR_OCCUPIED)
            else:
                window_dbscan_colors.append(COLOR_UNOCCUPIED)

        # ── Save snapshot ─────────────────────────────────────────────────────
        window_starts.append(window_start)
        window_ends.append(window_end)
        timestamps.append(dates[i])
        kmeans_colors.append(window_kmeans_colors)
        kmeans_accuracy.append(np.mean(np.array(kmeans_predictions) == window_truth))
        kmeans_silhouette.append(compute_silhouette(window_features, kmeans_labels))
        dbscan_colors.append(window_dbscan_colors)
        dbscan_accuracy.append(np.mean(np.array(dbscan_predictions) == window_truth))
        dbscan_silhouette.append(compute_silhouette(window_features, dbscan_labels))

        if len(window_starts) % 20 == 0:
            print(f"  Snapshot {len(window_starts)} | record {i} / {n}")

    print(f"Done. {len(window_starts)} snapshots ready.")

    return (
        window_starts, window_ends, timestamps,
        kmeans_colors, kmeans_accuracy, kmeans_silhouette,
        dbscan_colors, dbscan_accuracy, dbscan_silhouette,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build and save the animation
# ─────────────────────────────────────────────────────────────────────────────

def style_axis(ax, title, xlim, ylim):
    """Apply the dark-theme styling to a single subplot axes."""
    ax.set_facecolor(COLOR_PANEL)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("CO₂ (ppm)",   color=COLOR_TEXT, fontsize=11)
    ax.set_ylabel("Light (lux)", color=COLOR_TEXT, fontsize=11)
    ax.set_title(title, color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.tick_params(colors=COLOR_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, alpha=0.6)


def build_animation(co2, light, y_true, dates, snapshots):
    """
    Build and save the side-by-side KMeans / DBSCAN cluster animation.

    The figure has two layers per panel:
      - Static background: all records plotted once in faint grey so the
        viewer can always see the full data shape behind the active window.
      - Active window scatter: updated each frame to show the STREAM_WINDOW
        most-recent points, colored by their cluster prediction.

    Frame pacing is controlled by DATA_HOLD_FRAMES: each snapshot is displayed
    for that many frames before advancing, effectively slowing the animation
    without changing the underlying clustering logic.

    Args:
        co2       (np.ndarray): raw CO2 values (used for axis coordinates)
        light     (np.ndarray): raw Light values
        y_true    (np.ndarray): ground-truth labels (unused here, kept for signature parity)
        dates     (list[str]):  timestamps (unused here, kept for signature parity)
        snapshots (tuple):      output of precompute_checkpoints()
    """

    (window_starts, window_ends, timestamps,
     kmeans_colors, kmeans_accuracy, kmeans_silhouette,
     dbscan_colors, dbscan_accuracy, dbscan_silhouette) = snapshots

    fig, (ax_km, ax_db) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLOR_BG)

    # Add 3% padding around the data range so edge points aren't clipped
    x_pad = (co2.max() - co2.min()) * 0.03
    y_pad = (light.max() - light.min()) * 0.03
    xlim  = (co2.min() - x_pad,   co2.max() + x_pad)
    ylim  = (light.min() - y_pad, light.max() + y_pad)

    style_axis(ax_km, "Streaming K-Means  (k=2)", xlim, ylim)
    style_axis(ax_db, f"Streaming DBSCAN  (ε={DBSCAN_EPS}, min={DBSCAN_MIN_SAMPLES})", xlim, ylim)

    # Static background — all points faint grey, drawn once and never updated
    ax_km.scatter(co2, light, c=COLOR_HISTORY, s=4, alpha=0.3, linewidths=0, zorder=1)
    ax_db.scatter(co2, light, c=COLOR_HISTORY, s=4, alpha=0.3, linewidths=0, zorder=1)

    # Dynamic scatter objects — positions and colors are replaced each frame
    scatter_kmeans = ax_km.scatter([], [], s=10, linewidths=0, zorder=3)
    scatter_dbscan = ax_db.scatter([], [], s=10, linewidths=0, zorder=3)

    # Styled bounding boxes for the in-plot metric badges
    acc_badge = dict(boxstyle="round,pad=0.4", facecolor="#00FF9C18", edgecolor="#00FF9C66", linewidth=1.2)
    sil_badge = dict(boxstyle="round,pad=0.4", facecolor="#4CC9F018", edgecolor="#4CC9F066", linewidth=1.2)

    # Accuracy text overlays — top-right corner of each panel
    text_kmeans_accuracy = ax_km.text(0.98, 0.97, "", transform=ax_km.transAxes,
                             ha="right", va="top", color="#00FF9C",
                             fontsize=10, fontweight="bold", bbox=acc_badge)
    text_dbscan_accuracy = ax_db.text(0.98, 0.97, "", transform=ax_db.transAxes,
                             ha="right", va="top", color="#00FF9C",
                             fontsize=10, fontweight="bold", bbox=acc_badge)

    # Silhouette score overlays — just below the accuracy badges
    text_kmeans_silhouette = ax_km.text(0.98, 0.85, "", transform=ax_km.transAxes,
                             ha="right", va="top", color="#4CC9F0",
                             fontsize=10, fontweight="bold", bbox=sil_badge)
    text_dbscan_silhouette = ax_db.text(0.98, 0.85, "", transform=ax_db.transAxes,
                             ha="right", va="top", color="#4CC9F0",
                             fontsize=10, fontweight="bold", bbox=sil_badge)

    # Timestamp / window range shown at the bottom of the figure
    timestamp_text = fig.text(0.5, 0.01, "", ha="center", color=COLOR_TEXT, fontsize=9)

    legend_handles = [
        mpatches.Patch(color=COLOR_OCCUPIED,   label="Occupied"),
        mpatches.Patch(color=COLOR_UNOCCUPIED, label="Unoccupied"),
        mpatches.Patch(color=COLOR_NOISE,      label="Noise (DBSCAN only)"),
        mpatches.Patch(color=COLOR_HISTORY,    label="Outside window"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, facecolor=COLOR_PANEL, edgecolor=COLOR_GRID,
               labelcolor=COLOR_TEXT, framealpha=0.9, bbox_to_anchor=(0.5, 0.04))

    fig.suptitle("Occupancy Detection — Sliding Window Clustering",
                 color=COLOR_TEXT, fontsize=14, fontweight="bold")
    # Reserve space at the bottom for the legend and timestamp
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    def update(frame_idx):
        """
        Called by FuncAnimation for every frame.

        Converts the raw frame index to a snapshot index by integer-dividing
        by DATA_HOLD_FRAMES, so each snapshot is held for that many frames
        before the animation advances to the next one.
        """
        snapshot_index = frame_idx // DATA_HOLD_FRAMES

        ws = window_starts[snapshot_index]
        we = window_ends[snapshot_index]

        # Slice the raw (un-normalized) coordinates for the active window
        window_co2   = co2[ws : we + 1]
        window_light = light[ws : we + 1]

        # Update scatter positions and colors for both panels
        scatter_kmeans.set_offsets(np.column_stack([window_co2, window_light]))
        scatter_kmeans.set_color(kmeans_colors[snapshot_index])

        scatter_dbscan.set_offsets(np.column_stack([window_co2, window_light]))
        scatter_dbscan.set_color(dbscan_colors[snapshot_index])

        # Refresh accuracy badges
        text_kmeans_accuracy.set_text(f"Accuracy  {kmeans_accuracy[snapshot_index]:.1%}")
        text_dbscan_accuracy.set_text(f"Accuracy  {dbscan_accuracy[snapshot_index]:.1%}")

        # Refresh silhouette badges — show N/A when fewer than 2 clusters exist
        current_kmeans_silhouette = kmeans_silhouette[snapshot_index]
        current_dbscan_silhouette = dbscan_silhouette[snapshot_index]
        text_kmeans_silhouette.set_text(f"Silhouette  {current_kmeans_silhouette:.3f}" if current_kmeans_silhouette is not None else "Silhouette  N/A")
        text_dbscan_silhouette.set_text(f"Silhouette  {current_dbscan_silhouette:.3f}" if current_dbscan_silhouette is not None else "Silhouette  N/A")

        # Bottom timestamp: absolute record number, window index range, and datetime
        timestamp_text.set_text(f"Record {we + 1}  |  Window [{ws} – {we}]  |  {timestamps[snapshot_index]}")

        # Return all modified artists so blit=True only redraws what changed
        return scatter_kmeans, scatter_dbscan, text_kmeans_accuracy, text_dbscan_accuracy, text_kmeans_silhouette, text_dbscan_silhouette, timestamp_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(window_starts) * DATA_HOLD_FRAMES,
        interval=1000 // 60,  # target 60fps
        blit=True,
    )

    print(f"Rendering and saving to {SAVE_AS}...")
    # Use FFMpegWriter for MP4 (requires ffmpeg), PillowWriter for GIF
    writer = animation.FFMpegWriter(fps=60, bitrate=4000) if SAVE_AS.endswith(".mp4") else animation.PillowWriter(fps=60)
    ani.save(SAVE_AS, writer=writer, dpi=120, savefig_kwargs={"facecolor": COLOR_BG})
    print(f"Saved → {SAVE_AS}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    co2, light, y_true, dates = load_data(DATA_FILE)
    print(f"Loaded {len(co2)} records.")

    snapshots = precompute_checkpoints(co2, light, y_true, dates)
    build_animation(co2, light, y_true, dates, snapshots)
