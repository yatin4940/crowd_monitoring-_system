"""
Crowd Analysis Module — Modules 4 & 5 concepts.

Provides higher-level crowd-behaviour analytics on top of the raw
tracking data:

  1. K-Means spatial clustering of person centroids
     → finds natural crowd sub-groups within a frame.  (Module 4, S3)

  2. KNN-based density classification of each zone
     → labels zones as Low / Moderate / High / Critical.  (Module 5, S2)

  3. PCA crowd spread index
     → single number summarising how spread-out the crowd is.  (Module 5, S3)

  4. Naive Bayes simple crowd-state classifier
     → overall scene label: Safe / Caution / Danger.  (Module 5, S1)

These analytics are computed every N frames (configurable) to keep
CPU usage low; the most recent result is re-used in between.
"""

import numpy as np
from collections import Counter

from config import ANALYSIS_INTERVAL_FRAMES, DENSITY_KNN_K


# ---------------------------------------------------------------------------
# 1. K-Means spatial clustering  (Module 4 — Image Clustering)
# ---------------------------------------------------------------------------

def kmeans_crowd_clusters(centroids, n_clusters=3, max_iter=20):
    """
    Groups person centroids into spatial clusters using a plain NumPy
    implementation of K-Means.  Returns (labels, cluster_centres) where
    labels[i] is the cluster index for centroids[i].

    Falls back to a single cluster when there are fewer points than
    n_clusters (avoids divide-by-zero and degenerate cases).

    Parameters
    ----------
    centroids  : list of (x, y) tuples
    n_clusters : desired number of groups
    max_iter   : maximum iterations before giving up

    Returns
    -------
    labels         : list[int]  — cluster index per centroid
    cluster_centres: list[(float, float)]  — (x, y) of each cluster centre
    """
    if len(centroids) == 0:
        return [], []

    n_clusters = min(n_clusters, len(centroids))
    pts = np.array(centroids, dtype=np.float32)

    # Random initialisation — pick k unique points as starting centres.
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(pts), size=n_clusters, replace=False)
    centres = pts[idx].copy()

    labels = np.zeros(len(pts), dtype=int)

    for _ in range(max_iter):
        # Assignment step — each point goes to its nearest centre.
        diffs   = pts[:, np.newaxis, :] - centres[np.newaxis, :, :]   # (N, K, 2)
        dists   = np.linalg.norm(diffs, axis=2)                        # (N, K)
        new_labels = np.argmin(dists, axis=1)

        # Update step — recompute centres as the mean of assigned points.
        new_centres = np.array([
            pts[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centres[k]
            for k in range(n_clusters)
        ])

        if np.allclose(new_centres, centres, atol=1.0):
            labels = new_labels
            centres = new_centres
            break

        labels  = new_labels
        centres = new_centres

    return labels.tolist(), [(float(c[0]), float(c[1])) for c in centres]


# ---------------------------------------------------------------------------
# 2. KNN density classification  (Module 5 — Image Classification)
# ---------------------------------------------------------------------------

# Training data: (count_in_zone, density_label)
# These representative samples encode domain knowledge about what
# crowding levels mean in a typical indoor venue.
_KNN_TRAIN = [
    (0,  "Empty"),
    (1,  "Low"),
    (2,  "Low"),
    (3,  "Moderate"),
    (4,  "Moderate"),
    (5,  "Moderate"),
    (6,  "High"),
    (7,  "High"),
    (8,  "Critical"),
    (9,  "Critical"),
    (10, "Critical"),
    (11, "Critical"),
    (12, "Critical"),
]


def knn_density_label(count, k=DENSITY_KNN_K):
    """
    Classifies a zone's person count as a density label using K-Nearest
    Neighbours on a 1-D feature (the count itself).

    Parameters
    ----------
    count : int — number of people currently in the zone
    k     : int — neighbours to consider

    Returns
    -------
    str — one of "Empty", "Low", "Moderate", "High", "Critical"
    """
    train_counts = [x[0] for x in _KNN_TRAIN]
    train_labels = [x[1] for x in _KNN_TRAIN]

    # Euclidean distance on 1-D feature.
    distances = [abs(count - tc) for tc in train_counts]
    # Indices of the k smallest distances.
    k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
    k_labels  = [train_labels[i] for i in k_indices]

    # Majority vote.
    most_common = Counter(k_labels).most_common(1)[0][0]
    return most_common


# ---------------------------------------------------------------------------
# 3. PCA crowd-spread index  (Module 5 — Dimensionality Reduction)
# ---------------------------------------------------------------------------

def pca_spread_index(centroids):
    """
    Uses PCA to compute a single crowd-spread index.

    The principal components of the centroid point cloud are found.
    The spread index is the ratio of the largest eigenvalue to the
    total variance (sum of both eigenvalues).  A value near 1.0 means
    the crowd is stretched along a single line; near 0.5 means it is
    roughly circular / evenly spread.

    Returns 0.0 if fewer than 2 people are present.
    """
    if len(centroids) < 2:
        return 0.0

    pts = np.array(centroids, dtype=np.float64)
    pts -= pts.mean(axis=0)  # centre the data

    cov = np.cov(pts.T)      # 2x2 covariance matrix
    if cov.ndim < 2:
        # Only one unique point after centering.
        return 0.0

    eigenvalues, _ = np.linalg.eigh(cov)
    total = eigenvalues.sum()
    if total < 1e-9:
        return 0.0

    spread = float(eigenvalues.max() / total)
    return round(spread, 3)


# ---------------------------------------------------------------------------
# 4. Naive Bayes scene-level classifier  (Module 5 — Image Classification)
# ---------------------------------------------------------------------------

# Learned conditional probabilities (hand-tuned priors based on domain
# knowledge rather than a real dataset, which is acceptable for a
# demonstration system).
#
# P(state | features) is estimated using:
#   P(total_inside in bucket | state) * P(n_overcrowded_zones | state) * P(state)
#
# States: "Safe", "Caution", "Danger"

_NB_PRIORS = {"Safe": 0.5, "Caution": 0.3, "Danger": 0.2}

# P( total_inside bucket | state )
# Buckets: "low" (<5), "medium" (5-15), "high" (>15)
_NB_P_COUNT = {
    "Safe":    {"low": 0.70, "medium": 0.25, "high": 0.05},
    "Caution": {"low": 0.20, "medium": 0.60, "high": 0.20},
    "Danger":  {"low": 0.05, "medium": 0.25, "high": 0.70},
}

# P( overcrowded_zones bucket | state )
# Buckets: "none" (0), "some" (1-2), "many" (>2)
_NB_P_ZONES = {
    "Safe":    {"none": 0.90, "some": 0.08, "many": 0.02},
    "Caution": {"none": 0.30, "some": 0.55, "many": 0.15},
    "Danger":  {"none": 0.05, "some": 0.30, "many": 0.65},
}


def _count_bucket(total_inside):
    if total_inside < 5:
        return "low"
    elif total_inside <= 15:
        return "medium"
    return "high"


def _zone_bucket(n_overcrowded):
    if n_overcrowded == 0:
        return "none"
    elif n_overcrowded <= 2:
        return "some"
    return "many"


def naive_bayes_scene_state(total_inside, n_overcrowded_zones):
    """
    Classifies the overall crowd scene as Safe / Caution / Danger using
    a simple Naive Bayes model.

    Parameters
    ----------
    total_inside        : int — current estimate of people inside
    n_overcrowded_zones : int — how many zones are currently over capacity

    Returns
    -------
    str — "Safe", "Caution", or "Danger"
    """
    cb = _count_bucket(total_inside)
    zb = _zone_bucket(n_overcrowded_zones)

    scores = {}
    for state, prior in _NB_PRIORS.items():
        score = prior
        score *= _NB_P_COUNT[state].get(cb, 0.01)
        score *= _NB_P_ZONES[state].get(zb, 0.01)
        scores[state] = score

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class CrowdAnalyzer:
    """
    Wraps all four analytics methods and caches their results so the
    computation only runs every ANALYSIS_INTERVAL_FRAMES frames.
    """

    def __init__(self, interval=ANALYSIS_INTERVAL_FRAMES):
        self.interval     = interval
        self._frame_count = 0
        self._last_result = {}

    def update(self, tracked_objects, zone_status, total_inside):
        """
        Parameters
        ----------
        tracked_objects : list[TrackedObject]
        zone_status     : dict returned by ZoneManager.update()
        total_inside    : int from LineCounter.inside

        Returns
        -------
        dict with keys:
            "clusters"      : list[(float, float)]  — cluster centres
            "n_clusters"    : int
            "density_labels": dict[zone_name -> str]
            "spread_index"  : float
            "scene_state"   : str
        """
        self._frame_count += 1
        if self._frame_count % self.interval != 0 and self._last_result:
            return self._last_result

        centroids = [obj.centroid for obj in tracked_objects]

        # 1. Spatial clustering
        _, cluster_centres = kmeans_crowd_clusters(centroids, n_clusters=3)

        # 2. Per-zone density label via KNN
        density_labels = {}
        for name, info in zone_status.items():
            density_labels[name] = knn_density_label(info["count"])

        # 3. PCA spread index
        spread = pca_spread_index(centroids)

        # 4. Naive Bayes scene classification
        n_over = sum(1 for info in zone_status.values() if info["overcrowded"])
        scene  = naive_bayes_scene_state(total_inside, n_over)

        self._last_result = {
            "clusters"       : cluster_centres,
            "n_clusters"     : len(cluster_centres),
            "density_labels" : density_labels,
            "spread_index"   : spread,
            "scene_state"    : scene,
        }
        return self._last_result
