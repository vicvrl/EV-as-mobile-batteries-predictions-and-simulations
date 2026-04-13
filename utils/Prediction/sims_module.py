from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def _cyclical_time_features(ts: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(ts)
    hour = ts.dt.hour.astype(int)
    month = ts.dt.month.astype(int)
    weekday = ts.dt.weekday.astype(int)

    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    })


def _mode_or_nan(values: pd.Series):
    if len(values) == 0:
        return np.nan
    vc = values.value_counts(dropna=True)
    return vc.index[0] if len(vc) else np.nan


@dataclass
class SimSConfig:
    m: int = 50

    ts_col: str = "plug_in_datetime"
    user_col: str = "user_id"
    place_col: str = "place"

    dest_col: str = "next_dest"
    cons_col: str = "next_CBS"
    dur_col: str = "connected_duration"

    duration_log1p: bool = False


class SimilarSessionsModel:
    """
    Similar Sessions with cosine similarity.
    Inputs: user_id, current_place, ts (hour/month/weekday derived)
    Outputs: next_destination (mode), next_consumption (mean), session_duration (mean)
    """

    def __init__(self, cfg: SimSConfig):
        self.cfg = cfg
        self.ohe: Optional[OneHotEncoder] = None
        self.hist_df: Optional[pd.DataFrame] = None
        self._X_all: Optional[np.ndarray] = None

        self._cat_cols = [cfg.user_col, cfg.place_col]
        self._num_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"]

    def fit(self, df_hist: pd.DataFrame) -> "SimilarSessionsModel":
        cfg = self.cfg
        df = df_hist.copy()
        df[cfg.ts_col] = pd.to_datetime(df[cfg.ts_col])

        cyc = _cyclical_time_features(df[cfg.ts_col])
        df = pd.concat([df.reset_index(drop=True), cyc.reset_index(drop=True)], axis=1)

        if cfg.duration_log1p:
            df[cfg.dur_col] = np.log1p(df[cfg.dur_col].astype(float))

        self.hist_df = df

        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(df[self._cat_cols])

        X_cat = self.ohe.transform(df[self._cat_cols])
        X_num = df[self._num_cols].to_numpy(dtype=float)
        self._X_all = np.hstack([X_cat, X_num])

        return self

    def _features_from_state(self, user_id: Any, current_place: Any, ts: Any) -> np.ndarray:
        if self.ohe is None:
            raise RuntimeError("Model not fitted.")

        ts = pd.to_datetime(ts)
        df_state = pd.DataFrame({
            self.cfg.user_col: [user_id],
            self.cfg.place_col: [current_place],
            self.cfg.ts_col: [ts],
        })
        cyc = _cyclical_time_features(df_state[self.cfg.ts_col])
        df_state = pd.concat([df_state, cyc], axis=1)

        X_cat = self.ohe.transform(df_state[self._cat_cols])
        X_num = df_state[self._num_cols].to_numpy(dtype=float)
        return np.hstack([X_cat, X_num])

    def predict_state(self, user_id: Any, current_place: Any, ts: Any) -> Dict[str, Any]:
        """
        Usage: rule-based charging recommendation.
        """
        if self.hist_df is None or self._X_all is None:
            raise RuntimeError("Model not fitted.")

        cfg = self.cfg
        ts = pd.to_datetime(ts)

        mask = (self.hist_df[cfg.ts_col] < ts)
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            return {cfg.dest_col: np.nan, cfg.cons_col: np.nan, cfg.dur_col: np.nan}

        X_hist = self._X_all[idx]
        y_hist = self.hist_df.iloc[idx]

        X_target = self._features_from_state(user_id, current_place, ts)
        sims = cosine_similarity(X_target, X_hist).ravel()

        m = min(cfg.m, len(sims))
        top_local = np.argpartition(-sims, kth=m - 1)[:m]
        top_rows = y_hist.iloc[top_local]

        pred_dest = _mode_or_nan(top_rows[cfg.dest_col])
        pred_cons = float(np.mean(top_rows[cfg.cons_col].astype(float)))

        pred_dur = float(np.mean(top_rows[cfg.dur_col].astype(float)))
        if cfg.duration_log1p:
            pred_dur = float(np.expm1(pred_dur))

        return {cfg.dest_col: pred_dest, cfg.cons_col: pred_cons, cfg.dur_col: pred_dur}

    def predict_dataframe(self, df_test: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        preds = []
        for _, r in df_test.iterrows():
            preds.append(self.predict_state(
                user_id=r[cfg.user_col],
                current_place=r[cfg.place_col],
                ts=r[cfg.ts_col],
            ))
        return pd.DataFrame(preds, index=df_test.index)
