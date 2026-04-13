from __future__ import annotations

from typing import List, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from sims_module import SimSConfig, SimilarSessionsModel, smape


def split_train_test(
    df: pd.DataFrame,
    ts_col: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)

    train = df[df[ts_col] < test_start_dt].sort_values(ts_col)
    test = df[(df[ts_col] >= test_start_dt) & (df[ts_col] < test_end_dt)].sort_values(ts_col)
    return train, test


def _compute_metrics(
    df_test: pd.DataFrame,
    dest_pred: np.ndarray,
    cons_pred: np.ndarray,
    dur_pred: np.ndarray,
    cfg: SimSConfig,
    m: int,
    n_train: int,
) -> Dict[str, Any]:
    acc = float((df_test[cfg.dest_col].astype("object") == pd.Series(dest_pred, index=df_test.index).astype("object")).mean())

    y_true_cons = df_test[cfg.cons_col].astype(float).to_numpy()
    y_true_dur = df_test[cfg.dur_col].astype(float).to_numpy()

    return {
        "m": int(m),
        "destination_accuracy": acc,
        "consumption_mae": float(mean_absolute_error(y_true_cons, cons_pred)),
        "consumption_smape": smape(y_true_cons, cons_pred),
        "duration_mae": float(mean_absolute_error(y_true_dur, dur_pred)),
        "duration_smape": smape(y_true_dur, dur_pred),
        "n_test": int(len(df_test)),
        "n_train": int(n_train),
    }


def _chunked_top_indices(
    X_test: np.ndarray,
    X_hist_full: np.ndarray,
    cutoffs: np.ndarray,
    m_max: int,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    For each test row i, compute cosine similarity against X_hist_full[:cutoffs[i]]
    and return the indices of the top-min(m_max, cutoff) most similar training rows.

    Memory is controlled by processing test rows in chunks of `chunk_size`.
    Within each chunk, rows are further sub-grouped by their cutoff so we only
    build the portion of the history matrix that is actually needed.
    """
    n_test = len(X_test)
    top_indices = np.empty(n_test, dtype=object)

    for chunk_start in range(0, n_test, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_test)
        chunk_slice = slice(chunk_start, chunk_end)

        chunk_cutoffs = cutoffs[chunk_slice]
        X_q_chunk = X_test[chunk_slice]

        unique_cutoffs, inverse = np.unique(chunk_cutoffs, return_inverse=True)

        for uc_idx, cutoff in enumerate(unique_cutoffs):
            local_rows = np.where(inverse == uc_idx)[0]
            global_rows = local_rows + chunk_start

            if cutoff == 0:
                for ri in global_rows:
                    top_indices[ri] = np.array([], dtype=int)
                continue

            X_hist = X_hist_full[:cutoff]
            X_q = X_q_chunk[local_rows]

            sims = cosine_similarity(X_q, X_hist)

            m_eff = min(m_max, cutoff)
            top_part = np.argpartition(-sims, kth=m_eff - 1, axis=1)[:, :m_eff]
            row_range = np.arange(len(local_rows))[:, None]
            sorted_local = np.argsort(-sims[row_range, top_part], axis=1)
            top_sorted = top_part[row_range, sorted_local]

            for batch_i, ri in enumerate(global_rows):
                top_indices[ri] = top_sorted[batch_i]

        if (chunk_start // chunk_size) % 20 == 0:
            pct = 100 * chunk_end / n_test
            print(f"  similarity: {chunk_end}/{n_test} rows ({pct:.0f}%)", flush=True)

    return top_indices


def sweep_and_collect_predictions(
    df: pd.DataFrame,
    m_values: List[int],
    test_start: str,
    test_end: str,
    base_cfg: SimSConfig,
    sim_chunk_size: int = 256,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = split_train_test(df, base_cfg.ts_col, test_start, test_end)
    m_values_sorted = sorted(m_values, reverse=True)
    m_max = m_values_sorted[0]

    print(f"Fitting model once (m={m_max} used for feature construction)...")
    cfg_max = SimSConfig(**{**base_cfg.__dict__, "m": m_max})
    model = SimilarSessionsModel(cfg_max).fit(df_train)

    ts_col = base_cfg.ts_col
    user_col = base_cfg.user_col
    place_col = base_cfg.place_col

    print("Building test feature matrix (vectorized)...")
    from sims_module import _cyclical_time_features

    df_test_proc = df_test.copy()
    df_test_proc[ts_col] = pd.to_datetime(df_test_proc[ts_col])
    cyc_test = _cyclical_time_features(df_test_proc[ts_col])
    df_test_proc = pd.concat([df_test_proc.reset_index(drop=True), cyc_test.reset_index(drop=True)], axis=1)

    cat_cols = [user_col, place_col]
    num_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"]

    X_test_cat = model.ohe.transform(df_test_proc[cat_cols])
    X_test_num = df_test_proc[num_cols].to_numpy(dtype=float)
    X_test = np.hstack([X_test_cat, X_test_num])

    train_ts = model.hist_df[ts_col].to_numpy()
    test_ts = df_test_proc[ts_col].to_numpy()
    cutoffs = np.searchsorted(train_ts, test_ts, side="left")

    top_indices = _chunked_top_indices(
        X_test, model._X_all, cutoffs, m_max, chunk_size=sim_chunk_size
    )

    hist_df = model.hist_df
    n_test = len(df_test)

    results_rows = []
    predictions_df = df_test.copy()

    dest_col = base_cfg.dest_col
    cons_col = base_cfg.cons_col
    dur_col = base_cfg.dur_col

    for m in m_values:
        print(f"Aggregating predictions for m={m}...")
        dest_preds = []
        cons_preds = []
        dur_preds  = []

        for i in range(n_test):
            idx = top_indices[i]
            if len(idx) == 0:
                dest_preds.append(np.nan)
                cons_preds.append(np.nan)
                dur_preds.append(np.nan)
                continue

            top_m = idx[:min(m, len(idx))]
            top_rows = hist_df.iloc[top_m]

            dest_preds.append(_mode_or_nan(top_rows[dest_col]))
            cons_arr = top_rows[cons_col].astype(float).to_numpy()
            dur_arr  = top_rows[dur_col].astype(float).to_numpy()
            cons_preds.append(float(np.mean(cons_arr)))

            if base_cfg.duration_log1p:
                dur_preds.append(float(np.expm1(np.mean(dur_arr))))
            else:
                dur_preds.append(float(np.mean(dur_arr)))

        dest_preds_arr = np.array(dest_preds, dtype=object)
        cons_preds_arr = np.array(cons_preds, dtype=float)
        dur_preds_arr  = np.array(dur_preds,  dtype=float)

        predictions_df[f"{dest_col}_pred_m{m}"] = dest_preds_arr
        predictions_df[f"{cons_col}_pred_m{m}"] = cons_preds_arr
        predictions_df[f"{dur_col}_pred_m{m}"]  = dur_preds_arr

        results_rows.append(_compute_metrics(
            df_test, dest_preds_arr, cons_preds_arr, dur_preds_arr,
            base_cfg, m, len(df_train),
        ))

    results_df = pd.DataFrame(results_rows).sort_values("m").reset_index(drop=True)
    return results_df, predictions_df


def _mode_or_nan(values: pd.Series):
    if len(values) == 0:
        return np.nan
    vc = values.value_counts(dropna=True)
    return vc.index[0] if len(vc) else np.nan


if __name__ == "__main__":
    df = pd.read_csv("data/charging_sessions.csv")

    m_values = [5, 10, 20, 50]
    test_start = "2020-01-06"
    test_end   = "2020-02-03"

    base_cfg = SimSConfig()

    results, predictions = sweep_and_collect_predictions(
        df, m_values, test_start, test_end, base_cfg,
        sim_chunk_size=256,
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    metrics_path = os.path.join("results", "sims_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
    print(f"Saved sweep metrics to {metrics_path}")

    preds_path = os.path.join("output", "sims_test_predictions_all_m.csv")
    predictions.to_csv(preds_path, index=False)
    print(f"Saved all predictions to {preds_path}")