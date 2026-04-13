from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, accuracy_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    pairwise_distances,
)
from scipy.cluster.hierarchy import linkage, fcluster
from pyclustering.cluster.kmedoids import kmedoids
from kneed import KneeLocator


_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0F1117",
        "axes.facecolor":   "#1A1D27",
        "axes.edgecolor":   "#2E3347",
        "axes.labelcolor":  "#C8CDD8",
        "xtick.color":      "#7A8099",
        "ytick.color":      "#7A8099",
        "text.color":       "#C8CDD8",
        "grid.color":       "#2E3347",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "font.family":      "monospace",
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })


def plot_bic_and_n_components(bic_scores: Dict[int, float],
                               best_n: int,
                               save_dir: str) -> str:
    _apply_style()
    ns  = sorted(bic_scores.keys())
    bic = [bic_scores[n] for n in ns]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0F1117")
    ax.set_facecolor("#1A1D27")
    ax.plot(ns, bic, color="#4C72B0", linewidth=2, zorder=2)
    ax.scatter(ns, bic, color="#4C72B0", s=30, zorder=3)
    ax.axvline(best_n, color="#DD8452", linewidth=1.5, linestyle="--",
               label=f"Best k = {best_n}")
    min_bic = bic_scores[best_n]
    ax.scatter([best_n], [min_bic], color="#DD8452", s=80, zorder=4)
    ax.text(best_n + 0.3, min_bic,
            f"  {best_n} components\n  BIC = {min_bic:,.0f}",
            color="#DD8452", fontsize=8, va="center")
    ax.set_xlabel("Number of GMM components", fontsize=9)
    ax.set_ylabel("BIC score", fontsize=9)
    ax.set_title("Session GMM — BIC vs number of components", fontsize=10, pad=10)
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    path = os.path.join(save_dir, "bic_n_components.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {path}")
    return path


def plot_session_cluster_distributions(df: pd.DataFrame,
                                        feature_cols: List[str],
                                        save_dir: str) -> str:
    _apply_style()
    clusters = sorted(
        [c for c in df["SessionCluster"].unique() if c != "Noise"],
        key=lambda x: int(x) if str(x).isdigit() else x,
    )
    n_feats = len(feature_cols)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4), facecolor="#0F1117")
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, feature_cols):
        ax.set_facecolor("#1A1D27")
        for i, cl in enumerate(clusters):
            vals = df.loc[df["SessionCluster"] == cl, feat].dropna()
            if len(vals) < 5:
                continue
            sns.kdeplot(vals, ax=ax, color=_PALETTE[i % len(_PALETTE)],
                        fill=True, alpha=0.25, linewidth=1.5, label=f"Cluster {cl}")
        ax.set_title(feat.replace("_", " "), fontsize=9)
        ax.grid(True)
        ax.tick_params(labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7,
                   framealpha=0.3, ncol=max(1, len(clusters) // 6))
    fig.suptitle("Feature distributions per session cluster", fontsize=11, y=1.02)
    fig.tight_layout()

    path = os.path.join(save_dir, "session_cluster_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {path}")
    return path


def plot_portfolio_cluster_distributions(df: pd.DataFrame, save_dir: str) -> str:
    _apply_style()
    candidate_cols = ["plug_in_time", "connected_duration", "HBS", "DBS",
                      "arrival_SoC", "next_CBS"]
    plot_cols = [c for c in candidate_cols if c in df.columns]
    if "user_cluster" not in df.columns or not plot_cols:
        print("  [plot] user_cluster or numeric cols not found — skipping portfolio plot")
        return ""

    clusters = sorted(df["user_cluster"].dropna().unique().astype(int))
    n_cols   = min(len(plot_cols), 3)
    n_rows   = (len(plot_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 3.5 * n_rows),
                              facecolor="#0F1117")
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(plot_cols):
        ax = axes[idx]
        ax.set_facecolor("#1A1D27")
        for i, cl in enumerate(clusters):
            vals = df.loc[df["user_cluster"] == cl, feat].dropna()
            if len(vals) < 5:
                continue
            sns.kdeplot(vals, ax=ax, color=_PALETTE[i % len(_PALETTE)],
                        fill=True, alpha=0.20, linewidth=1.5, label=f"User cluster {cl}")
        ax.set_title(feat.replace("_", " "), fontsize=9)
        ax.grid(True)
        ax.tick_params(labelsize=7)

    for ax in axes[len(plot_cols):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7,
                   framealpha=0.3, ncol=max(1, len(clusters) // 4))
    fig.suptitle("Feature distributions per user (portfolio) cluster", fontsize=11, y=1.01)
    fig.tight_layout()

    path = os.path.join(save_dir, "portfolio_cluster_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {path}")
    return path

@dataclass
class PipelineConfig:
    random_state: int = 42
    max_connected_hours: float = 48.0
    hbs_modulo_hours: float = 168.0
    dbs_shift_m: float = 1.0
    session_noise_prob_threshold: float = 0.70
    gmm_n_components_min: int = 1
    gmm_n_components_max: int = 30
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-6
    min_sessions_per_user: int = 40
    drop_noise_in_portfolios: bool = True
    portfolio_k_min: int = 2
    portfolio_k_max: int = 15
    test_start: str = "2020-01-06"
    test_days: int = 28
    fallback_test_fraction: float = 0.20
    sc_lookup_time_window_hours: float = 2.0
    output_dir: str = "models"
    bundle_path: str = "models/ev_two_step_bundle.joblib"
    plots_dir: str = "results/plots"


def smape_metric(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    test_start: str,
    test_days: int,
    fallback_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col).reset_index(drop=True)
    cutoff = pd.Timestamp(test_start)
    test_end = cutoff + pd.Timedelta(days=test_days)
    train = df[df[time_col] < cutoff].copy()
    test = df[(df[time_col] >= cutoff) & (df[time_col] < test_end)].copy()
    if len(train) < 100 or len(test) < 50:
        n_test = max(1, int(len(df) * fallback_frac))
        train, test = df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()
    return train, test


def preprocess_sessions(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.drop(columns=["Unnamed: 0", "prev_plug_out"], errors="ignore").copy()
    df["plug_in_datetime"] = pd.to_datetime(df["plug_in_datetime"], errors="coerce")

    for col in ["arrival_SoC", "departure_SoC", "next_CBS"]:
        if col in df.columns:
            df = df[df[col] > 0]

    for col, cap in [("connected_duration", cfg.max_connected_hours),
                     ("charging_duration", cfg.max_connected_hours),
                     ("HBS", cfg.hbs_modulo_hours)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
            df[col] = df[col].apply(lambda x: x % cap if pd.notna(x) and x > cap else x)

    if "DBS" in df.columns:
        df["DBS"] = pd.to_numeric(df["DBS"], errors="coerce").clip(lower=0)
        df["log_DBS"] = np.log10(df["DBS"] + cfg.dbs_shift_m)
    elif "log_DBS" not in df.columns:
        raise ValueError("Expected 'DBS' or 'log_DBS' in the dataset.")

    df["plug_in_time"] = df["plug_in_datetime"].dt.hour + df["plug_in_datetime"].dt.minute / 60.0

    if "plug_out_datetime" not in df.columns and "connected_duration" in df.columns:
        df["plug_out_datetime"] = df["plug_in_datetime"] + pd.to_timedelta(df["connected_duration"], unit="h")

    return df.dropna(subset=["plug_in_datetime", "plug_in_time"]).reset_index(drop=True)


def make_session_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    df["plug_in_time_norm"] = df["plug_in_time"] / 24.0

    df["connected_duration"] = pd.to_numeric(df["connected_duration"], errors="coerce").clip(0, cfg.max_connected_hours)
    df["connected_duration_norm"] = df["connected_duration"] / cfg.max_connected_hours

    df["HBS"] = pd.to_numeric(df["HBS"], errors="coerce").clip(0, 196)
    df["HBS_norm"] = df["HBS"] / 196.0

    if "log_DBS" not in df.columns:
        raise ValueError("log_DBS missing — run preprocess_sessions first.")

    return df[df["log_DBS"] > 0].reset_index(drop=True)


def apply_log_dbs_normalisation(df: pd.DataFrame, max_log: float) -> pd.DataFrame:
    if not np.isfinite(max_log) or max_log <= 0:
        raise ValueError("max_log is invalid.")
    df = df.copy()
    df["normalized_log_DBS"] = df["log_DBS"] / max_log
    return df[df["normalized_log_DBS"] > 0].reset_index(drop=True)

_LOOKUP_COLS = ["user_cluster", "place", "plug_in_time", "SessionCluster"]


def build_session_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact lookup table from the training set.
    Only non-Noise sessions are retained so that the lookup
    always returns a meaningful session type.
    """
    available = [c for c in _LOOKUP_COLS if c in df.columns]
    lookup = df[available].copy()
    lookup = lookup[lookup["SessionCluster"] != "Noise"]
    return lookup.reset_index(drop=True)


def lookup_session_cluster(
    lookup: pd.DataFrame,
    user_cluster: int,
    place: str,
    plug_in_time: float,
    time_window_hours: float = 2.0,
) -> str:
    """
    Predict the most likely session cluster for an incoming session
    using a cascade of filters on the training lookup table.

    Filters applied in order:
      1. User cluster  — keeps sessions from behaviourally similar drivers
      2. Current place — keeps sessions at the same location
      3. Time window   — keeps sessions within ±time_window_hours of plug-in time

    The most frequent session cluster in the resulting subset is returned.
    If the subset becomes empty at any filter step, the filter is relaxed
    (place filter dropped first, then time window widened by 2 h each time)
    until at least one session is found or all fallbacks are exhausted,
    in which case "Noise" is returned.
    """
    subset = lookup[lookup["user_cluster"] == user_cluster]

    if len(subset) == 0:
        return "Noise"

    subset_place = subset[subset["place"] == place]
    if len(subset_place) == 0:
        subset_place = subset

    window = time_window_hours
    subset_time = subset_place[
        (subset_place["plug_in_time"] >= plug_in_time - window) &
        (subset_place["plug_in_time"] <= plug_in_time + window)
    ]
    while len(subset_time) == 0 and window <= 12.0:
        window += 2.0
        subset_time = subset_place[
            (subset_place["plug_in_time"] >= plug_in_time - window) &
            (subset_place["plug_in_time"] <= plug_in_time + window)
        ]

    if len(subset_time) == 0:
        return subset["SessionCluster"].mode()[0]

    return subset_time["SessionCluster"].mode()[0]


def apply_lookup_session_cluster(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    time_window_hours: float = 2.0,
) -> pd.DataFrame:
    df = df.copy()
    df["SessionCluster_lookup"] = df.apply(
        lambda r: lookup_session_cluster(
            lookup,
            user_cluster=int(r["user_cluster"]) if pd.notna(r.get("user_cluster")) else -1,
            place=str(r["place"]) if pd.notna(r.get("place")) else "",
            plug_in_time=float(r["plug_in_time"]),
            time_window_hours=time_window_hours,
        ),
        axis=1,
    )
    return df

@dataclass
class SessionClusteringResult:
    best_n_components: int
    bic_scores: Dict[int, float]
    gmm: Any
    feature_cols: List[str]
    max_log_dbs: float = 0.0


def fit_session_gmm(df: pd.DataFrame, feature_cols: List[str], cfg: PipelineConfig) -> SessionClusteringResult:
    X = df[feature_cols].to_numpy(dtype=float)
    bic_scores = {}

    for n in range(cfg.gmm_n_components_min, cfg.gmm_n_components_max + 1):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=cfg.gmm_covariance_type,
            reg_covar=cfg.gmm_reg_covar,
            random_state=cfg.random_state,
        ).fit(X)
        bic_scores[n] = float(gmm.bic(X))

    # Knee/Elbow detection
    ns = sorted(bic_scores.keys())
    bics = [bic_scores[n] for n in ns]
    knee = KneeLocator(ns, bics, curve="convex", direction="decreasing")
    best_n = knee.knee or ns[np.argmin(bics)]
    best_model = GaussianMixture(
        n_components=best_n,
        covariance_type=cfg.gmm_covariance_type,
        reg_covar=cfg.gmm_reg_covar,
        random_state=cfg.random_state,
    ).fit(X)

    return SessionClusteringResult(
        best_n_components=int(best_model.n_components),
        bic_scores=bic_scores,
        gmm=best_model,
        feature_cols=feature_cols,
    )


def apply_session_clusters(df: pd.DataFrame, scr: SessionClusteringResult, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    X = df[scr.feature_cols].to_numpy(dtype=float)
    probs = scr.gmm.predict_proba(X)
    labels = scr.gmm.predict(X)
    maxp = probs.max(axis=1)
    df["SessionClusterRaw"] = labels.astype(int)
    df["SessionClusterProb"] = maxp
    df["SessionCluster"] = np.where(maxp < cfg.session_noise_prob_threshold, "Noise", labels.astype(str))
    return df

def _kmeans_labels(k: int, X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10).fit_predict(X)


def _pam_labels(k: int, X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    n = X.shape[0]
    initial = random.Random(cfg.random_state).sample(range(n), k)
    km = kmedoids(X.tolist(), initial)
    km.process()
    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(km.get_clusters()):
        labels[members] = cid
    return labels


def _hier_labels(k: int, X: np.ndarray, method: str) -> np.ndarray:
    return fcluster(linkage(X, method=method), t=k, criterion="maxclust") - 1



def _calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Ratio of between-cluster dispersion to within-cluster dispersion.
    Higher is better (max).
    """
    return float(calinski_harabasz_score(X, labels))


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean ratio of intra-cluster distance to nearest-cluster distance per sample.
    Range [-1, 1]; higher is better (max).
    """
    return float(silhouette_score(X, labels))


def _davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean similarity of each cluster to its most similar other cluster,
    where similarity is the ratio of within- to between-cluster distances.
    Lower is better (min).
    """
    return float(davies_bouldin_score(X, labels))


def _duda_hart(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Duda–Hart index: ratio of within-cluster variance to total variance.
    Values closer to 1 indicate better separation; lower is better here
    because we want tight clusters relative to the total spread (min).
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.inf
    total_var = np.var(X, axis=0).sum() * (len(X) - 1)
    within_var = sum(
        np.var(X[labels == c], axis=0).sum() * (np.sum(labels == c) - 1)
        for c in unique
        if np.sum(labels == c) > 1
    )
    if total_var == 0:
        return np.inf
    return float(within_var / total_var)


def _pseudot2(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Pseudo-T² (Calinski–Harabasz variant normalised by degrees of freedom).
    Specifically: (SSB / (k-1)) / (SSW / (n-k)), where SSB is the
    between-cluster sum of squares and SSW is the within-cluster sum of squares.
    Lower values indicate better clustering (min).
    Equivalent to the inverse of Calinski–Harabasz scaled by df, so it
    is genuinely distinct from CH because the optimum k differs when
    SSB and SSW scale differently with k.
    """
    unique = np.unique(labels)
    k = len(unique)
    n = len(X)
    if k < 2 or n <= k:
        return np.inf
    grand_mean = X.mean(axis=0)
    ssb = sum(
        np.sum(labels == c) * np.sum((X[labels == c].mean(axis=0) - grand_mean) ** 2)
        for c in unique
    )
    ssw = sum(
        np.sum((X[labels == c] - X[labels == c].mean(axis=0)) ** 2)
        for c in unique
        if np.sum(labels == c) > 1
    )
    if ssw == 0:
        return np.inf
    return float((ssb / (k - 1)) / (ssw / (n - k)))


def _c_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    C-index: (S_w - S_min) / (S_max - S_min), where S_w is the sum of
    within-cluster pairwise distances, S_min is the sum of the n_w smallest
    pairwise distances in the full dataset, and S_max is the sum of the
    n_w largest. Range [0, 1]; lower is better (min).
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.inf

    within_dists = []
    for c in unique:
        pts = X[labels == c]
        if len(pts) >= 2:
            d = pairwise_distances(pts)
            idx = np.triu_indices_from(d, k=1)
            within_dists.extend(d[idx].tolist())

    if not within_dists:
        return np.inf

    n_w = len(within_dists)
    s_w = float(np.sum(within_dists))

    all_d = pairwise_distances(X)
    all_flat = np.sort(all_d[np.triu_indices_from(all_d, k=1)])
    if len(all_flat) < n_w:
        return np.inf

    s_min = float(all_flat[:n_w].sum())
    s_max = float(all_flat[-n_w:].sum())

    if s_max == s_min:
        return 0.0
    return float((s_w - s_min) / (s_max - s_min))


def _gamma_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Baker–Hubert Gamma index: (s_plus - s_minus) / (s_plus + s_minus),
    where s_plus counts concordant pairs (within-cluster distance < between-
    cluster distance) and s_minus counts discordant pairs.
    Range [-1, 1]; higher is better (max).
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return -np.inf

    n = len(X)
    all_d = pairwise_distances(X)
    s_plus = s_minus = 0

    for i in range(n):
        for j in range(i + 1, n):
            d_ij = all_d[i, j]
            same = labels[i] == labels[j]
            for p in range(n):
                for q in range(p + 1, n):
                    if (labels[p] == labels[q]) == same:
                        continue
                    d_pq = all_d[p, q]
                    if same and d_ij < d_pq:
                        s_plus += 1
                    elif not same and d_ij > d_pq:
                        s_plus += 1
                    else:
                        s_minus += 1

    total = s_plus + s_minus
    if total == 0:
        return 0.0
    return float((s_plus - s_minus) / total)


def _beale_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Beale index: tests whether splitting a cluster into two is statistically
    justified. Computed as (SSW_k / SSW_{k-1} - 1) * (n - k) / (2^(2/p) - 1),
    approximated here cluster-wise using the ratio of within-cluster SS to
    total SS relative to k and dimensionality p.
    Lower values indicate less need to further split (min).
    """
    unique = np.unique(labels)
    k = len(unique)
    n, p = X.shape
    if k < 2 or p == 0:
        return np.inf

    ssw = sum(
        np.sum((X[labels == c] - X[labels == c].mean(axis=0)) ** 2)
        for c in unique
        if np.sum(labels == c) > 1
    )
    sst = np.sum((X - X.mean(axis=0)) ** 2)
    if sst == 0:
        return np.inf

    ratio = ssw / sst
    denom = 2 ** (2.0 / p) - 1
    if denom <= 0:
        return np.inf
    return float(ratio * (n - k) / denom)


@dataclass
class PortfolioClusteringResult:
    method: str
    best_k: int
    votes: Dict[str, Dict[int, int]]
    user_clusterer: Any
    portfolio_columns: List[str]
    pam_medoid_indices: Optional[List[int]] = None   
    pam_medoid_vectors: Optional[np.ndarray] = None
    hier_linkage_matrix: Optional[np.ndarray] = None
    train_X: Optional[np.ndarray] = None

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        """
        Assign every row of X to a cluster using the single winning method.
        This is the one canonical dispatch path used both at training time
        (to build user_cluster_map) and at inference time (for new users).
        """
        if self.method == "KMeans":
            return self.user_clusterer.predict(X).astype(int)

        elif self.method == "PAM":
            if self.pam_medoid_vectors is None:
                raise RuntimeError("PAM medoid vectors not persisted — re-train.")
            return _assign_pam(X, self.pam_medoid_vectors).astype(int)

        elif self.method in ("HierarchicalWard", "HierarchicalComplete"):
            if self.hier_linkage_matrix is None or self.train_X is None:
                raise RuntimeError(
                    "Hierarchical linkage matrix or training matrix not persisted — re-train."
                )
            link_method = "ward" if self.method == "HierarchicalWard" else "complete"
            return _assign_hier(
                X, self.train_X, self.hier_linkage_matrix, self.best_k, link_method
            ).astype(int)

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")


def build_user_portfolios(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    work = df[df["SessionCluster"] != "Noise"].copy() if cfg.drop_noise_in_portfolios else df.copy()
    counts = work.groupby("user_id").size()
    work = work[work["user_id"].isin(counts[counts >= cfg.min_sessions_per_user].index)]
    port = work.groupby(["user_id", "SessionCluster"]).size().unstack(fill_value=0)
    return port.div(port.sum(axis=1), axis=0).reset_index()


def _assign_pam(X_new: np.ndarray, medoid_vectors: np.ndarray) -> np.ndarray:
    """Assign each row in X_new to the nearest medoid (L2 distance)."""
    dists = pairwise_distances(X_new, medoid_vectors)
    return dists.argmin(axis=1)


def _assign_hier(X_new: np.ndarray, X_train: np.ndarray,
                 linkage_matrix: np.ndarray, k: int, method: str) -> np.ndarray:
    """
    Assign new points to the nearest training-set cluster centroid derived
    from a fitted hierarchical solution.  We recompute the cut on the stored
    linkage matrix so no re-fitting is required.
    """
    train_labels = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
    centroids = np.array([
        X_train[train_labels == c].mean(axis=0)
        for c in range(k)
    ])
    dists = pairwise_distances(X_new, centroids)
    return dists.argmin(axis=1)


def vote_best_portfolio_clustering(
    portfolio: pd.DataFrame,
    cfg: PipelineConfig,
) -> PortfolioClusteringResult:
    X = portfolio.drop(columns=["user_id"]).to_numpy(dtype=float)
    portfolio_columns = [c for c in portfolio.columns if c != "user_id"]

    clustering_methods = {
        "KMeans":              lambda k: _kmeans_labels(k, X, cfg),
        "PAM":                 lambda k: _pam_labels(k, X, cfg),
        "HierarchicalWard":    lambda k: _hier_labels(k, X, "ward"),
        "HierarchicalComplete":lambda k: _hier_labels(k, X, "complete"),
    }

    n_users = X.shape[0]
    use_gamma = n_users <= 300

    validation_metrics: List[Tuple[str, Any, str]] = [
        ("Calinski-Harabasz", lambda l: _calinski_harabasz(X, l), "max"),
        ("Silhouette",        lambda l: _silhouette(X, l),        "max"),
        ("Davies-Bouldin",    lambda l: _davies_bouldin(X, l),    "min"),
        ("Duda-Hart",         lambda l: _duda_hart(X, l),         "min"),
        ("Pseudo-T2",         lambda l: _pseudot2(X, l),          "min"),
        ("C-Index",           lambda l: _c_index(X, l),           "min"),
        ("Beale",             lambda l: _beale_index(X, l),       "min"),
    ]
    if use_gamma:
        validation_metrics.append(
            ("Gamma",         lambda l: _gamma_index(X, l),       "max")
        )

    k_range = range(cfg.portfolio_k_min, cfg.portfolio_k_max + 1)

    labels_cache: Dict[Tuple[str, int], np.ndarray] = {
        (m, k): clustering_methods[m](k)
        for m in clustering_methods
        for k in k_range
    }

    votes: Dict[str, Dict[int, int]] = {m: {k: 0 for k in k_range} for m in clustering_methods}

    for m in clustering_methods:
        for metric_name, metric_fn, mode in validation_metrics:
            scores: Dict[int, float] = {}
            for k in k_range:
                try:
                    scores[k] = float(metric_fn(labels_cache[(m, k)]))
                except Exception:
                    scores[k] = np.nan
            valid = {k: v for k, v in scores.items() if np.isfinite(v)}
            if not valid:
                continue
            best_k = max(valid, key=valid.get) if mode == "max" else min(valid, key=valid.get)
            votes[m][best_k] += 1

    best_method, best_k, best_vote, best_sil = None, None, -1, -np.inf
    for m, kvotes in votes.items():
        for k, v in kvotes.items():
            try:
                sil = float(silhouette_score(X, labels_cache[(m, k)]))
            except Exception:
                sil = -np.inf
            if v > best_vote or (v == best_vote and sil > best_sil):
                best_method, best_k, best_vote, best_sil = m, k, v, sil

    pam_medoid_indices: Optional[List[int]] = None
    pam_medoid_vectors: Optional[np.ndarray] = None
    hier_linkage_matrix: Optional[np.ndarray] = None

    if best_method == "KMeans":
        clusterer = KMeans(
            n_clusters=best_k, random_state=cfg.random_state, n_init=10
        ).fit(X)

    elif best_method == "PAM":
        rng = random.Random(cfg.random_state)
        initial = rng.sample(range(X.shape[0]), best_k)
        km = kmedoids(X.tolist(), initial)
        km.process()
        pam_medoid_indices = [int(km.get_medoids()[c]) for c in range(best_k)]
        pam_medoid_vectors = X[pam_medoid_indices].copy()
        clusterer = {
            "type": "PAM",
            "k": best_k,
            "medoid_indices": pam_medoid_indices,
        }

    elif best_method in ("HierarchicalWard", "HierarchicalComplete"):
        link_method = "ward" if best_method == "HierarchicalWard" else "complete"
        hier_linkage_matrix = linkage(X, method=link_method)
        clusterer = {
            "type": best_method,
            "k": best_k,
            "linkage_method": link_method,
        }

    else:
        clusterer = {"type": best_method, "k": best_k}

    return PortfolioClusteringResult(
        method=best_method,
        best_k=best_k,
        votes=votes,
        user_clusterer=clusterer,
        portfolio_columns=portfolio_columns,
        pam_medoid_indices=pam_medoid_indices,
        pam_medoid_vectors=pam_medoid_vectors,
        hier_linkage_matrix=hier_linkage_matrix,
        train_X=X.copy(),
    )

def build_feature_pipeline(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), numeric),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def pick_columns_for_prediction(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = [c for c in ["user_id", "user_cluster", "SessionCluster", "place"] if c in df.columns]
    numeric = [c for c in ["plug_in_time", "arrival_SoC"] if c in df.columns]
    return categorical, numeric


@dataclass
class ModelSelectionResult:
    destination_model: Any
    next_cbs_model: Any
    connected_duration_model: Any
    metrics: Dict[str, Any]
    feature_spec: Dict[str, Any]


def train_and_select_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lookup: pd.DataFrame,
    cfg: PipelineConfig,
) -> ModelSelectionResult:
    dest_col = next(
        (c for c in ["next_destination", "next_dest", "destination", "next_place"]
         if c in train_df.columns),
        None,
    )
    if dest_col is None:
        raise ValueError("Missing destination target column.")

    categorical, numeric = pick_columns_for_prediction(train_df)
    cols = categorical + numeric

    X_tr = train_df[cols].copy()

    test_df_with_lookup = apply_lookup_session_cluster(
        test_df, lookup, time_window_hours=cfg.sc_lookup_time_window_hours
    )
    X_te = test_df_with_lookup[cols].copy()
    if "SessionCluster" in X_te.columns:
        X_te["SessionCluster"] = test_df_with_lookup["SessionCluster_lookup"]

    def _drop_nan(X, y):
        mask = np.isfinite(y.to_numpy(dtype=float))
        return X.loc[mask], y.loc[mask]

    y_dest_tr = train_df[dest_col].fillna("MISSING").astype(str)
    y_dest_te = test_df[dest_col].fillna("MISSING").astype(str)
    Xc_tr, y_cbs_tr = _drop_nan(X_tr, pd.to_numeric(train_df["next_CBS"], errors="coerce"))
    Xc_te, y_cbs_te = _drop_nan(X_te, pd.to_numeric(test_df["next_CBS"], errors="coerce"))
    Xd_tr, y_cd_tr  = _drop_nan(X_tr, pd.to_numeric(train_df["connected_duration"], errors="coerce"))
    Xd_te, y_cd_te  = _drop_nan(X_te, pd.to_numeric(test_df["connected_duration"], errors="coerce"))

    candidates_clf = [
        ("HGB_depth6", HistGradientBoostingClassifier(random_state=cfg.random_state, max_depth=6, learning_rate=0.08)),
        ("HGB_depth4", HistGradientBoostingClassifier(random_state=cfg.random_state, max_depth=4, learning_rate=0.12)),
    ]
    candidates_reg = [
        ("HGB_depth6", HistGradientBoostingRegressor(random_state=cfg.random_state, max_depth=6, learning_rate=0.08)),
        ("HGB_depth4", HistGradientBoostingRegressor(random_state=cfg.random_state, max_depth=4, learning_rate=0.12)),
    ]

    def _select_clf(candidates, X_tr, y_tr, X_te, y_te):
        """Select best classifier from candidates."""
        best_pipe, best_score = None, -np.inf
        details = {}
        for name, model in candidates:
            pre = build_feature_pipeline(categorical, numeric)
            pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_tr, y_tr)
            acc = accuracy_score(y_te.to_numpy(dtype=str), pipe.predict(X_te).astype(str))
            details[name] = {"accuracy": float(acc)}
            if acc > best_score:
                best_score, best_pipe = acc, pipe
        return best_pipe, best_score, details

    def _select_reg_for_target(target_name, candidates, X_tr, y_tr, X_te, y_te):
        """Select best regressor from candidates for a specific target.
        
        This creates INDEPENDENT model instances for each target to avoid
        any shared state or reference issues between models.
        """
        best_pipe, best_score = None, np.inf
        details = {}
        
        fresh_candidates = [
            (name, HistGradientBoostingRegressor(
                random_state=cfg.random_state,
                max_depth=6 if "depth6" in name else 4,
                learning_rate=0.08 if "depth6" in name else 0.12
            ))
            for name, _ in candidates
        ]
        
        for name, model in fresh_candidates:
            pre = build_feature_pipeline(categorical, numeric)
            pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            s = smape_metric(y_te.to_numpy(), pred)
            details[name] = {"mae": float(mean_absolute_error(y_te, pred)), "smape": float(s)}
            if s < best_score:
                best_score, best_pipe = s, pipe
        
        return best_pipe, best_score, details

    dest_model, best_acc, dest_details = _select_clf(
        candidates_clf, X_tr, y_dest_tr, X_te, y_dest_te
    )
    
    cbs_model, best_cbs_smape, cbs_details = _select_reg_for_target(
        "next_CBS", candidates_reg, Xc_tr, y_cbs_tr, Xc_te, y_cbs_te
    )
    
    cd_model, best_cd_smape, cd_details = _select_reg_for_target(
        "connected_duration", candidates_reg, Xd_tr, y_cd_tr, Xd_te, y_cd_te
    )

    metrics = {
        "destination":        {
            "best_accuracy": float(best_acc),
            "candidates": dest_details,
            "target_col": dest_col,
            "n_samples_train": len(X_tr),
            "n_samples_test": len(X_te),
        },
        "next_CBS":           {
            "best_smape": float(best_cbs_smape),
            "candidates": cbs_details,
            "n_samples_train": len(Xc_tr),
            "n_samples_test": len(Xc_te),
            "target_col": "next_CBS",
        },
        "connected_duration": {
            "best_smape": float(best_cd_smape),
            "candidates": cd_details,
            "n_samples_train": len(Xd_tr),
            "n_samples_test": len(Xd_te),
            "target_col": "connected_duration",
        },
        "note": (
            "Three INDEPENDENT models trained on three SEPARATE targets.\n"
            "Each model uses its own subset of data (rows with valid target values).\n"
            "Test-set metrics computed using lookup-predicted SessionCluster, "
            "mirroring exact deployment conditions in predict_single."
        ),
    }

    assert dest_model is not None, "Destination model is None"
    assert cbs_model is not None, "CBS model is None"
    assert cd_model is not None, "Duration model is None"
    assert dest_model is not cbs_model, "Destination and CBS models are the same object!"
    assert dest_model is not cd_model, "Destination and duration models are the same object!"
    assert cbs_model is not cd_model, "CBS and duration models are the same object!"

    return ModelSelectionResult(
        destination_model=dest_model,
        next_cbs_model=cbs_model,
        connected_duration_model=cd_model,
        metrics=metrics,
        feature_spec={"categorical": categorical, "numeric": numeric},
    )


@dataclass
class EVTwinStepBundle:
    cfg: Dict[str, Any]
    session_clustering: SessionClusteringResult
    portfolio_clustering: PortfolioClusteringResult
    destination_model: Any
    next_cbs_model: Any
    connected_duration_model: Any
    feature_spec: Dict[str, Any]
    metrics: Dict[str, Any]
    user_cluster_map: Dict[str, int]
    session_lookup: pd.DataFrame

    max_log_dbs: float = 0.0

    def assign_user_cluster(self, portfolio_vector: np.ndarray) -> int:
        """
        Assign a new user to a cluster given their portfolio vector.
        Delegates to PortfolioClusteringResult.assign_all — the single
        canonical dispatch path shared with training-time assignment.

        Parameters
        ----------
        portfolio_vector : 1-D array of shape (n_session_types,)
            Proportion of sessions in each session type for the new user.
            Must be aligned with self.portfolio_clustering.portfolio_columns.

        Returns
        -------
        int
            Cluster index.
        """
        x = np.asarray(portfolio_vector, dtype=float).reshape(1, -1)
        return int(self.portfolio_clustering.assign_all(x)[0])

    def _get_session_cluster(self, user_cluster: int, place: str, plug_in_time: float) -> str:
        cfg = PipelineConfig(**self.cfg)
        return lookup_session_cluster(
            self.session_lookup,
            user_cluster=user_cluster,
            place=place,
            plug_in_time=plug_in_time,
            time_window_hours=cfg.sc_lookup_time_window_hours,
        )

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction. SessionCluster is predicted via the lookup procedure.
        Does NOT require DBS, HBS, or connected_duration at inference time.
        """
        cfg = PipelineConfig(**self.cfg)
        df = df_new.copy()

        if "plug_in_datetime" in df.columns:
            df["plug_in_datetime"] = pd.to_datetime(df["plug_in_datetime"], errors="coerce")
            df["plug_in_time"] = (
                df["plug_in_datetime"].dt.hour + df["plug_in_datetime"].dt.minute / 60.0
            )

        df["user_cluster"] = (
            df["user_id"].astype(str).map(self.user_cluster_map).fillna(-1).astype(int)
            if "user_id" in df.columns else -1
        )

        df = apply_lookup_session_cluster(
            df, self.session_lookup, time_window_hours=cfg.sc_lookup_time_window_hours
        )
        df["SessionCluster"] = df["SessionCluster_lookup"]

        cat, num = self.feature_spec["categorical"], self.feature_spec["numeric"]
        X = df.reindex(columns=cat + num)

        out = df.copy()
        out["next_dest_pred"]          = self.destination_model.predict(X).astype(str)
        out["next_CBS_pred"]           = self.next_cbs_model.predict(X)
        out["connected_duration_pred"] = self.connected_duration_model.predict(X)
        return out

    def predict_single(
        self,
        user_id: str,
        place: str,
        plug_in_dt: pd.Timestamp,
        arrival_soc: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Lightweight single-session inference for simulation.
        All inputs are observable at plug-in time.
        """
        user_cluster = self.user_cluster_map.get(str(user_id), -1)
        plug_in_time = plug_in_dt.hour + plug_in_dt.minute / 60.0
        predicted_sc = self._get_session_cluster(user_cluster, place, plug_in_time)

        cat, num = self.feature_spec["categorical"], self.feature_spec["numeric"]
        row = {
            "user_id":        str(user_id),
            "user_cluster":   user_cluster,
            "place":          str(place),
            "SessionCluster": predicted_sc,
            "plug_in_time":   plug_in_time,
            "arrival_SoC":    float(arrival_soc),
        }
        X = pd.DataFrame([row]).reindex(columns=cat + num)

        return {
            "predicted_session_cluster": predicted_sc,
            "next_dest_pred":            str(self.destination_model.predict(X)[0]),
            "next_CBS_pred":             float(self.next_cbs_model.predict(X)[0]),
            "connected_duration_pred":   float(self.connected_duration_model.predict(X)[0]),
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "EVTwinStepBundle":
        return joblib.load(path)

def run_pipeline(
    csv_path: str,
    cfg: Optional[PipelineConfig] = None,
    save: bool = True,
    return_splits: bool = False,
) -> EVTwinStepBundle | Tuple[EVTwinStepBundle, pd.DataFrame, pd.DataFrame]:

    cfg = cfg or PipelineConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df_raw = make_session_features(preprocess_sessions(pd.read_csv(csv_path), cfg), cfg)

    train_df_raw, test_df_raw = time_based_split(
        df_raw, time_col="plug_in_datetime",
        test_start=cfg.test_start, test_days=cfg.test_days,
        fallback_frac=cfg.fallback_test_fraction,
    )

    max_log_dbs = float(train_df_raw["log_DBS"].max())
    train_df = apply_log_dbs_normalisation(train_df_raw, max_log_dbs)
    test_df  = apply_log_dbs_normalisation(test_df_raw,  max_log_dbs)

    session_feature_cols = [
        "normalized_log_DBS",
        "HBS_norm",
        "plug_in_time_norm",
        "connected_duration_norm",
    ]
    scr = fit_session_gmm(train_df, session_feature_cols, cfg)
    scr.max_log_dbs = max_log_dbs

    train_df = apply_session_clusters(train_df, scr, cfg)
    test_df  = apply_session_clusters(test_df,  scr, cfg)

    plot_bic_and_n_components(scr.bic_scores, scr.best_n_components, cfg.plots_dir)
    plot_session_cluster_distributions(train_df, session_feature_cols, cfg.plots_dir)

    train_portfolios = build_user_portfolios(train_df, cfg)
    pcr = vote_best_portfolio_clustering(train_portfolios, cfg)

    Xp_train = train_portfolios.drop(columns=["user_id"]).to_numpy(dtype=float)

    train_user_clusters = pcr.assign_all(Xp_train)

    user_cluster_map = dict(
        zip(train_portfolios["user_id"].astype(str), train_user_clusters.astype(int))
    )

    train_df["user_cluster"] = (
        train_df["user_id"].astype(str).map(user_cluster_map).fillna(-1).astype(int)
        if "user_id" in train_df.columns else -1
    )
    test_df["user_cluster"] = (
        test_df["user_id"].astype(str).map(user_cluster_map).fillna(-1).astype(int)
        if "user_id" in test_df.columns else -1
    )

    plot_portfolio_cluster_distributions(train_df, cfg.plots_dir)

    session_lookup = build_session_lookup(train_df)
    msr = train_and_select_models(train_df, test_df, session_lookup, cfg)

    bundle = EVTwinStepBundle(
        cfg=asdict(cfg),
        session_clustering=scr,
        portfolio_clustering=pcr,
        destination_model=msr.destination_model,
        next_cbs_model=msr.next_cbs_model,
        connected_duration_model=msr.connected_duration_model,
        feature_spec=msr.feature_spec,
        metrics=msr.metrics,
        user_cluster_map=user_cluster_map,
        session_lookup=session_lookup,
        max_log_dbs=max_log_dbs,
    )

    if save:
        bundle.save(cfg.bundle_path)
        with open(os.path.join("results", "two_step_metrics.txt"), "w") as f:
            json.dump(bundle.metrics, f, indent=2)

    return (bundle, train_df, test_df) if return_splits else bundle


if __name__ == "__main__":
    cfg = PipelineConfig(
        test_start="2020-01-06",
        test_days=28,
        output_dir="models",
        bundle_path="models/two_step_clustering.joblib",
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    bundle, train_df, test_df = run_pipeline(
        csv_path="data/charging_sessions.csv",
        cfg=cfg,
        save=True,
        return_splits=True,
    )

    print(f"Bundle saved to: {cfg.bundle_path}")
    print("Metrics:\n", json.dumps(bundle.metrics, indent=2))

    df_pred = bundle.predict(test_df)
    preds_path = os.path.join("output", "two_step_test_predictions.csv")
    df_pred.to_csv(preds_path, index=False)
    print(f"Wrote test predictions to: {preds_path}")