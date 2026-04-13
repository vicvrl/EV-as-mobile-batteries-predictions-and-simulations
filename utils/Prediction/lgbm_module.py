from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[time_col])
    out["hour"] = ts.dt.hour.astype(int)
    out["month"] = ts.dt.month.astype(int)
    out["weekday"] = ts.dt.weekday.astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0)
    return out


class TargetTransform:
    def __init__(self, mode: Optional[str] = "log1p"):
        self.mode = mode

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.mode is None:
            return y
        if self.mode == "log1p":
            return np.log1p(np.clip(y, 0, None))
        raise ValueError(f"Unknown transform mode: {self.mode}")

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        y_t = np.asarray(y_t, dtype=float)
        if self.mode is None:
            return y_t
        if self.mode == "log1p":
            return np.expm1(y_t)
        raise ValueError(f"Unknown transform mode: {self.mode}")


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))


@dataclass
class SplitConfig:
    test_start: str
    test_days: int
    lookback_days: int
    val_days: int = 14


@dataclass
class TrainConfig:
    user_col: str = "user_id"
    place_col: str = "place"
    time_col: str = "plug_in_datetime"
    target_next_dest: str = "next_dest"
    target_next_CBS: str = "next_CBS"
    target_connected_duration: str = "connected_duration"
    categorical_cols: Tuple[str, ...] = ("user_id", "place")
    numeric_cols: Tuple[str, ...] = (
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "weekday_sin", "weekday_cos",
        "arrival_SoC",
    )
    CBS_transform: Optional[str] = "log1p"
    duration_transform: Optional[str] = "log1p"
    optuna_trials: int = 40


def time_based_splits(
    df: pd.DataFrame,
    time_col: str,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    test_start = pd.Timestamp(cfg.test_start)
    test_end = test_start + pd.Timedelta(days=cfg.test_days)
    train_end = test_start
    train_start = train_end - pd.Timedelta(days=cfg.lookback_days)
    val_start = train_end - pd.Timedelta(days=cfg.val_days)

    train = df[(df[time_col] >= train_start) & (df[time_col] < train_end)]
    test = df[(df[time_col] >= test_start) & (df[time_col] < test_end)]

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty train/test split. Check dates and data coverage.")

    tr = train[train[time_col] < val_start].copy()
    va = train[train[time_col] >= val_start].copy()

    if len(tr) == 0 or len(va) == 0:
        raise ValueError("Empty train/val split. Reduce val_days or increase lookback_days.")

    return tr, va, test.copy()


def make_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", ohe),
    ])
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[("cat", cat, categorical_cols), ("num", num, numeric_cols)],
        remainder="drop",
    )


def _suggest_params(trial: optuna.Trial, task: str) -> Dict[str, Any]:
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "objective": task,
    }
    return params


def _build_params(best: Dict[str, Any], task: str) -> Dict[str, Any]:
    defaults = {
        "n_estimators": 5000, "learning_rate": 0.01, "num_leaves": 31,
        "max_depth": -1, "min_child_samples": 20, "subsample": 1.0,
        "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0,
        "random_state": 42, "n_jobs": -1, "verbosity": -1, "objective": task,
    }
    defaults.update({k: best[k] for k in defaults if k in best})
    return defaults


def tune_and_train_classifier(
    X_tr, y_tr, X_va, y_va,
    preprocessor: ColumnTransformer,
    n_classes: int,
    n_trials: int = 40,
) -> Tuple[Pipeline, optuna.Study]:

    def objective(trial):
        params = _suggest_params(trial, "multiclass")
        params["num_class"] = n_classes
        prep = clone(preprocessor)
        X_tr_t = prep.fit_transform(X_tr)
        X_va_t = prep.transform(X_va)
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)], eval_metric="multi_logloss")
        return float(log_loss(y_va, clf.predict_proba(X_va_t), labels=list(range(n_classes))))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = _build_params(study.best_params, "multiclass")
    best["num_class"] = n_classes
    model = Pipeline([("prep", preprocessor), ("clf", lgb.LGBMClassifier(**best))])
    model.fit(X_tr, y_tr)
    return model, study


def tune_and_train_regressor(
    X_tr, y_tr, X_va, y_va,
    preprocessor: ColumnTransformer,
    y_transform: TargetTransform,
    n_trials: int = 40,
) -> Tuple[Pipeline, optuna.Study]:

    y_tr_t = y_transform.forward(y_tr)
    y_va_t = y_transform.forward(y_va)

    def objective(trial):
        params = _suggest_params(trial, "regression")
        prep = clone(preprocessor)
        X_tr_t = prep.fit_transform(X_tr)
        X_va_t = prep.transform(X_va)
        reg = lgb.LGBMRegressor(**params)
        reg.fit(X_tr_t, y_tr_t, eval_set=[(X_va_t, y_va_t)], eval_metric="l2")
        return float(mean_absolute_error(y_va, y_transform.inverse(reg.predict(X_va_t))))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = _build_params(study.best_params, "regression")
    model = Pipeline([
        ("prep", preprocessor),
        ("reg", TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**best),
            func=y_transform.forward,
            inverse_func=y_transform.inverse,
        )),
    ])
    model.fit(X_tr, y_tr)
    return model, study


def _make_time_row(ts: pd.Timestamp) -> Dict[str, float]:
    h, m, w = ts.hour, ts.month, ts.weekday()
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24.0),
        "hour_cos": np.cos(2 * np.pi * h / 24.0),
        "month_sin": np.sin(2 * np.pi * m / 12.0),
        "month_cos": np.cos(2 * np.pi * m / 12.0),
        "weekday_sin": np.sin(2 * np.pi * w / 7.0),
        "weekday_cos": np.cos(2 * np.pi * w / 7.0),
    }


@dataclass
class EVPredictionBundle:
    model_next_dest: Pipeline
    model_next_CBS: Pipeline
    model_connected_duration: Pipeline
    dest_idx_to_label: Dict[int, str]
    CBS_transform: TargetTransform
    duration_transform: TargetTransform
    categorical_cols: Tuple[str, ...] = ("user_id", "place")
    time_col: str = "plug_in_datetime"

    def _build_row(self, user_id: Any, place: Any, ts: Any, arrival_soc: float = 0.0) -> pd.DataFrame:
        ts = pd.Timestamp(ts)
        row = {self.categorical_cols[0]: str(user_id), self.categorical_cols[1]: str(place)}
        row.update(_make_time_row(ts))
        row["arrival_SoC"] = float(arrival_soc)
        return pd.DataFrame([row])

    def predict_next_dest(self, user_id, place, ts, arrival_soc: float = 0.0, top_k: int = 5) -> List[Dict]:
        X = self._build_row(user_id, place, ts, arrival_soc=arrival_soc)
        proba = self.model_next_dest.predict_proba(X)[0]
        idx = np.argsort(proba)[::-1][:top_k]
        return [{"next_dest": self.dest_idx_to_label[int(i)], "prob": float(proba[i])} for i in idx]

    def predict_next_CBS(self, user_id, place, ts, arrival_soc: float = 0.0) -> float:
        return max(0.0, float(self.model_next_CBS.predict(self._build_row(user_id, place, ts, arrival_soc=arrival_soc))[0]))

    def predict_connected_duration(self, user_id, place, ts, arrival_soc: float = 0.0) -> float:
        return max(0.0, float(self.model_connected_duration.predict(self._build_row(user_id, place, ts, arrival_soc=arrival_soc))[0]))

    def save(self, path: str) -> None:
        meta = {
            "lightgbm_version": getattr(lgb, "__version__", None),
            "sklearn_version": getattr(sklearn, "__version__", None),
            "optuna_version": getattr(optuna, "__version__", None),
        }
        joblib.dump({"bundle": self, "meta": meta}, path, compress=("gzip", 3))

    @staticmethod
    def load(path: str) -> "EVPredictionBundle":
        payload = joblib.load(path)
        if isinstance(payload, dict) and "bundle" in payload:
            meta = payload.get("meta", {})
            cur = {
                "lightgbm_version": getattr(lgb, "__version__", None),
                "sklearn_version": getattr(sklearn, "__version__", None),
                "optuna_version": getattr(optuna, "__version__", None),
            }
            mismatches = {k: (meta[k], cur[k]) for k in cur if meta.get(k) and meta[k] != cur[k]}
            if mismatches:
                warnings.warn(f"Version mismatch in loaded bundle: {mismatches}")
            return payload["bundle"]
        if isinstance(payload, EVPredictionBundle):
            warnings.warn("Loaded bundle in legacy format (no metadata).")
            return payload
        raise RuntimeError("Unrecognized bundle file format.")


def train_three_models(
    df: pd.DataFrame,
    split_cfg: SplitConfig,
    cfg: TrainConfig = TrainConfig(),
    save_path: str = "models/lgbm.joblib",
) -> Dict[str, Any]:

    df = add_time_features(df, time_col=cfg.time_col)

    required = [cfg.user_col, cfg.place_col, cfg.time_col,
                cfg.target_next_dest, cfg.target_next_CBS, cfg.target_connected_duration]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tr, va, te = time_based_splits(df, time_col=cfg.time_col, cfg=split_cfg)

    feature_cols = list(cfg.categorical_cols) + list(cfg.numeric_cols)
    X_tr, X_va, X_te = tr[feature_cols], va[feature_cols], te[feature_cols]

    preprocessor = make_preprocessor(list(cfg.categorical_cols), list(cfg.numeric_cols))

    classes = sorted(tr[cfg.target_next_dest].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    def encode_dest(s):
        return s.astype(str).map(class_to_idx)

    yA_tr = encode_dest(tr[cfg.target_next_dest])
    yA_va = encode_dest(va[cfg.target_next_dest])
    yA_te = encode_dest(te[cfg.target_next_dest])

    tr_mask, va_mask, te_mask = yA_tr.notna(), yA_va.notna(), yA_te.notna()
    if (~va_mask).any() or (~te_mask).any():
        warnings.warn(f"Dropped unseen destination labels: val={int((~va_mask).sum())}, test={int((~te_mask).sum())}")

    model_A, study_A = tune_and_train_classifier(
        X_tr[tr_mask], yA_tr[tr_mask].astype(int).to_numpy(),
        X_va[va_mask], yA_va[va_mask].astype(int).to_numpy(),
        preprocessor=preprocessor,
        n_classes=len(classes),
        n_trials=cfg.optuna_trials,
    )

    CBS_transform = TargetTransform(cfg.CBS_transform)
    model_B, study_B = tune_and_train_regressor(
        X_tr, tr[cfg.target_next_CBS].to_numpy(dtype=float),
        X_va, va[cfg.target_next_CBS].to_numpy(dtype=float),
        preprocessor=preprocessor,
        y_transform=CBS_transform,
        n_trials=cfg.optuna_trials,
    )

    duration_transform = TargetTransform(cfg.duration_transform)
    model_C, study_C = tune_and_train_regressor(
        X_tr, tr[cfg.target_connected_duration].to_numpy(dtype=float),
        X_va, va[cfg.target_connected_duration].to_numpy(dtype=float),
        preprocessor=preprocessor,
        y_transform=duration_transform,
        n_trials=cfg.optuna_trials,
    )

    yA_te_clean = yA_te[te_mask].astype(int).to_numpy()
    pred_A_masked = model_A.predict(X_te[te_mask])
    metrics_A = {
        "accuracy": float(accuracy_score(yA_te_clean, pred_A_masked)),
        "confusion_matrix": confusion_matrix(yA_te_clean, pred_A_masked).tolist(),
    }

    pred_A_full = np.full(len(X_te), fill_value=-1, dtype=int)
    pred_A_full[te_mask.to_numpy()] = pred_A_masked

    yB_te = te[cfg.target_next_CBS].to_numpy(dtype=float)
    pred_B = model_B.predict(X_te)
    metrics_B = {"MAE": float(mean_absolute_error(yB_te, pred_B)), "SMAPE": smape(yB_te, pred_B)}

    yC_te = te[cfg.target_connected_duration].to_numpy(dtype=float)
    pred_C = model_C.predict(X_te)
    metrics_C = {"MAE": float(mean_absolute_error(yC_te, pred_C)), "SMAPE": smape(yC_te, pred_C)}

    bundle = EVPredictionBundle(
        model_next_dest=model_A,
        model_next_CBS=model_B,
        model_connected_duration=model_C,
        dest_idx_to_label=idx_to_class,
        CBS_transform=CBS_transform,
        duration_transform=duration_transform,
        categorical_cols=cfg.categorical_cols,
        time_col=cfg.time_col,
    )
    bundle.save(save_path)

    return {
        "bundle_path": save_path,
        "df_test": te.reset_index(drop=True),
        "preds_test": {
            cfg.target_next_dest: [idx_to_class.get(p, "UNKNOWN") for p in pred_A_full],
            cfg.target_next_CBS: pred_B.tolist(),
            cfg.target_connected_duration: pred_C.tolist(),
        },
        "metrics_test": {
            "next_dest": metrics_A,
            "next_CBS": metrics_B,
            "connected_duration": metrics_C,
        },
        "studies": {"next_dest": study_A, "next_CBS": study_B, "connected_duration": study_C},
        "split_cfg": split_cfg,
    }