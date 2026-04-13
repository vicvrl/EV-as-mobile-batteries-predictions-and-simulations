from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, top_k_accuracy_score

from gmm_modules import PlacePGMM, personalize_weights_only, GMMDecisionBundle

K_GRID = [3, 4, 5, 6, 7, 8, 10, 12, 15]
N_INIT = 25
CV_SPLITS = 5
RANDOM_STATE = 42
MIN_PLACE_SESSIONS = 50
MIN_USER_SESSIONS = 30
N_EM_ITERS = 30
ALPHA_PRIOR = 1.0

DATA_PATH = Path("data/charging_sessions.csv")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
TEST_YEAR = 2020
TEST_DAYS = 28


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))))


def basic_cleaning(
    df: pd.DataFrame,
    time_col: str, duration_col: str, energy_col: str,
    dest_col: str, place_col: str, user_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col, duration_col, energy_col, dest_col, place_col, user_col])
    out[duration_col] = pd.to_numeric(out[duration_col], errors="coerce")
    out[energy_col] = pd.to_numeric(out[energy_col], errors="coerce")
    out = out.dropna(subset=[duration_col, energy_col])
    out = out[(out[duration_col] > 0) & (out[energy_col] >= 0)]
    for col in [dest_col, place_col, user_col]:
        out[col] = out[col].astype(str)
    return out.sort_values(time_col).reset_index(drop=True)


def time_split(df: pd.DataFrame, time_col: str, test_year: int, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    test_start = pd.Timestamp(year=test_year, month=1, day=6)
    test_end = test_start + pd.Timedelta(days=test_days)
    return (
        df[df[time_col] < test_start].copy(),
        df[(df[time_col] >= test_start) & (df[time_col] < test_end)].copy(),
    )


def train_population_models(
    train_df: pd.DataFrame,
    place_col: str, time_col: str, duration_col: str, energy_col: str, dest_col: str,
    n_components_grid: List[int], n_init: int, cv_splits: int, random_state: int,
    min_place_sessions: int = 50, verbose: bool = True,
) -> Dict[str, PlacePGMM]:
    place_models = {}
    for place, dfp in train_df.groupby(place_col):
        if len(dfp) < min_place_sessions:
            if verbose:
                print(f"[WARN] place={place} has {len(dfp)} sessions; skipping.")
            continue
        if verbose:
            print(f"\n--- Training P-GMM for place={place} (n={len(dfp)}) ---")
        pgmm = PlacePGMM.fit_from_df(
            df_place=dfp,
            n_components_grid=n_components_grid,
            n_init=n_init, random_state=random_state, cv_splits=cv_splits,
            time_col=time_col, duration_col=duration_col,
            energy_col=energy_col, dest_col=dest_col, verbose=verbose,
        )
        place_models[str(place)] = pgmm
        if verbose:
            print(f"place={place}: K={pgmm.n_components} | classes={len(pgmm.dest_classes)}")
    if not place_models:
        raise RuntimeError("No place models were trained.")
    return place_models


def train_individual_weights(
    place_models: Dict[str, PlacePGMM],
    train_df: pd.DataFrame,
    place_col: str, user_col: str, time_col: str, duration_col: str, energy_col: str,
    min_user_sessions: int = 30, n_em_iters: int = 30, alpha_prior: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    user_weights: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in place_models}
    for place, pgmm in place_models.items():
        dfp = train_df[train_df[place_col] == place]
        count = 0
        for user_id, dfu in dfp.groupby(user_col):
            if len(dfu) < min_user_sessions:
                continue
            user_weights[place][str(user_id)] = personalize_weights_only(
                pgmm=pgmm, df_user_place=dfu,
                time_col=time_col, duration_col=duration_col, energy_col=energy_col,
                n_em_iters=n_em_iters, alpha_prior=alpha_prior,
            )
            count += 1
        if verbose:
            print(f"[OK] place={place}: personalized users={count}")
    return user_weights


def predict_to_dataframe(
    bundle: GMMDecisionBundle,
    df_test: pd.DataFrame,
    time_col: str, place_col: str, user_col: str,
    use_personalized: bool,
    dest_pred_col: str, duration_pred_col: str, energy_pred_col: str,
) -> pd.DataFrame:
    df = df_test.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    dest_preds, dur_preds, en_preds = [], [], []
    for row in df.itertuples(index=False):
        out = bundle.predict(
            place=str(getattr(row, place_col)),
            arrival_time=getattr(row, time_col),
            user_id=str(getattr(row, user_col)) if use_personalized else None,
        )
        dest_preds.append(str(out["destination_pred"]))
        dur_preds.append(float(out["duration_pred"]))
        en_preds.append(float(out["energy_pred"]))
    df[dest_pred_col] = np.array(dest_preds, dtype=str)
    df[duration_pred_col] = np.array(dur_preds, dtype=float)
    df[energy_pred_col] = np.array(en_preds, dtype=float)
    return df


def evaluate_bundle(
    bundle: GMMDecisionBundle,
    df_test: pd.DataFrame,
    time_col: str, place_col: str, user_col: str,
    dest_col: str, duration_col: str, energy_col: str,
    use_personalized: bool = True,
    topk: Tuple[int, ...] = (2, 3),
) -> pd.DataFrame:
    df = df_test.copy().reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col])

    y_true = df[dest_col].astype(str).to_numpy()
    d_true = df[duration_col].astype(float).to_numpy()
    e_true = df[energy_col].astype(float).to_numpy()

    classes = sorted(pd.Series(y_true).unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    y_pred, d_pred, e_pred, proba_mat = [], [], [], []
    for row in df.itertuples(index=False):
        out = bundle.predict(
            place=str(getattr(row, place_col)),
            arrival_time=getattr(row, time_col),
            user_id=str(getattr(row, user_col)) if use_personalized else None,
        )
        y_pred.append(str(out["destination_pred"]))
        d_pred.append(float(out["duration_pred"]))
        e_pred.append(float(out["energy_pred"]))
        p = np.array([float(out.get("destination_proba", {}).get(c, 0.0)) for c in classes], dtype=float)
        s = p.sum()
        if s > 0:
            p /= s
        else:
            if out["destination_pred"] in cls_to_idx:
                p[cls_to_idx[out["destination_pred"]]] = 1.0
        proba_mat.append(p)

    y_pred_arr = np.array(y_pred, dtype=str)
    d_pred_arr = np.array(d_pred, dtype=float)
    e_pred_arr = np.array(e_pred, dtype=float)
    proba_arr = np.vstack(proba_mat) if proba_mat else np.zeros((0, len(classes)))

    def _row(scope, idx):
        yt, yp = y_true[idx], y_pred_arr[idx]
        dt, dp = d_true[idx], d_pred_arr[idx]
        et, ep = e_true[idx], e_pred_arr[idx]
        r = {
            "scope": scope,
            "n_samples": len(idx),
            "personalized": use_personalized,
            "dest_accuracy": float(accuracy_score(yt, yp)),
            "duration_mae": float(mean_absolute_error(dt, dp)),
            "duration_rmse": rmse(dt, dp),
            "duration_smape": smape(dt, dp),
            "energy_mae": float(mean_absolute_error(et, ep)),
            "energy_rmse": rmse(et, ep),
            "energy_smape": smape(et, ep),
        }
        return r

    all_idx = np.arange(len(df))
    overall = _row("overall", all_idx)

    if len(df) >= 2:
        y_true_idx = np.array([cls_to_idx.get(c, -1) for c in y_true], dtype=int)
        if (y_true_idx >= 0).all():
            for k in topk:
                if k < len(classes):
                    overall[f"dest_top{k}_accuracy"] = float(
                        top_k_accuracy_score(y_true_idx, proba_arr, k=k, labels=np.arange(len(classes)))
                    )

    rows = [overall]
    for place, dfp in df.groupby(place_col):
        rows.append(_row(f"place={place}", dfp.index.to_numpy()))

    return pd.DataFrame(rows)


def bundle_score(metrics_df: pd.DataFrame) -> Tuple[float, float]:
    overall = metrics_df.loc[metrics_df["scope"] == "overall"].iloc[0]
    return float(overall["dest_accuracy"]), -(float(overall["duration_rmse"]) + float(overall["energy_rmse"]))


def format_metrics_txt(
    metrics_pop: pd.DataFrame,
    metrics_ind: pd.DataFrame,
    best_name: str,
    score_pop: Tuple[float, float],
    score_ind: Tuple[float, float],
) -> str:
    lines = ["=" * 70, "GMM MODEL EVALUATION REPORT", "=" * 70,
             f"\nBest bundle: {best_name.upper()}",
             f"  Population   score: accuracy={score_pop[0]:.6f}, -(dur+energy RMSE)={score_pop[1]:.6f}",
             f"  Personalized score: accuracy={score_ind[0]:.6f}, -(dur+energy RMSE)={score_ind[1]:.6f}"]

    for label, mdf in [("POPULATION (P-GMM)", metrics_pop), ("PERSONALIZED (I-GMM)", metrics_ind)]:
        lines += ["\n" + "=" * 70, f"  {label}", "=" * 70]
        overall = mdf[mdf["scope"] == "overall"].iloc[0]
        lines.append(f"\n  [Overall]  n={int(overall['n_samples'])}")
        for k, v in overall.items():
            if k not in ("scope", "n_samples", "personalized"):
                lines.append(f"    {k:<25}: {v:.6f}" if isinstance(v, float) else f"    {k:<25}: {v}")
        for _, row in mdf[mdf["scope"] != "overall"].iterrows():
            lines.append(f"\n  {row['scope']}  (n={int(row['n_samples'])})")
            for k, v in row.items():
                if k not in ("scope", "n_samples", "personalized"):
                    lines.append(f"    {k:<25}: {v:.6f}" if isinstance(v, float) else f"    {k:<25}: {v}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main() -> None:
    time_col, duration_col, energy_col = "plug_in_datetime", "connected_duration", "next_CBS"
    dest_col, place_col, user_col = "next_dest", "place", "user_id"

    base_dir = Path(__file__).resolve().parent
    data_path = DATA_PATH if DATA_PATH.is_absolute() else base_dir / DATA_PATH
    out_dir = (OUTPUT_DIR if OUTPUT_DIR.is_absolute() else base_dir / OUTPUT_DIR)
    models_dir = (MODELS_DIR if MODELS_DIR.is_absolute() else base_dir / MODELS_DIR)
    results_dir = (RESULTS_DIR if RESULTS_DIR.is_absolute() else base_dir / RESULTS_DIR)
    for d in [out_dir, models_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found at: {data_path}")

    df = basic_cleaning(
        pd.read_csv(data_path),
        time_col=time_col, duration_col=duration_col, energy_col=energy_col,
        dest_col=dest_col, place_col=place_col, user_col=user_col,
    )
    train_df, test_df = time_split(df, time_col=time_col, test_year=TEST_YEAR, test_days=TEST_DAYS)

    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Empty split. Train={len(train_df)}, Test={len(test_df)}.")

    print(f"Train n={len(train_df)} | Test n={len(test_df)}")

    place_models = train_population_models(
        train_df, place_col=place_col, time_col=time_col,
        duration_col=duration_col, energy_col=energy_col, dest_col=dest_col,
        n_components_grid=K_GRID, n_init=N_INIT, cv_splits=CV_SPLITS,
        random_state=RANDOM_STATE, min_place_sessions=MIN_PLACE_SESSIONS,
    )

    bundle_pop = GMMDecisionBundle(place_models=place_models, user_weights={p: {} for p in place_models})
    bundle_pop.save((models_dir / "pgmm_bundle.joblib").as_posix())
    print(f"\nSaved P-GMM bundle: {models_dir / 'pgmm_bundle.joblib'}")

    metrics_pop = evaluate_bundle(
        bundle_pop, test_df, time_col=time_col, place_col=place_col, user_col=user_col,
        dest_col=dest_col, duration_col=duration_col, energy_col=energy_col,
        use_personalized=False,
    )

    user_weights = train_individual_weights(
        place_models, train_df, place_col=place_col, user_col=user_col,
        time_col=time_col, duration_col=duration_col, energy_col=energy_col,
        min_user_sessions=MIN_USER_SESSIONS, n_em_iters=N_EM_ITERS, alpha_prior=ALPHA_PRIOR,
    )

    bundle_ind = GMMDecisionBundle(place_models=place_models, user_weights=user_weights)
    bundle_ind.save((models_dir / "igmm_bundle.joblib").as_posix())
    print(f"Saved I-GMM bundle: {models_dir / 'igmm_bundle.joblib'}")

    metrics_ind = evaluate_bundle(
        bundle_ind, test_df, time_col=time_col, place_col=place_col, user_col=user_col,
        dest_col=dest_col, duration_col=duration_col, energy_col=energy_col,
        use_personalized=True,
    )

    score_pop = bundle_score(metrics_pop)
    score_ind = bundle_score(metrics_ind)
    best_name = "personalized" if score_ind > score_pop else "population"
    print(f"\nPopulation score: {score_pop} | Personalized score: {score_ind} | Best: {best_name}")

    for bundle, name, use_pers in [
        (bundle_pop, "pgmm", False),
        (bundle_ind, "igmm", True),
    ]:
        df_pred = predict_to_dataframe(
            bundle, test_df, time_col=time_col, place_col=place_col, user_col=user_col,
            use_personalized=use_pers,
            dest_pred_col=f"{dest_col}_pred",
            duration_pred_col=f"{duration_col}_pred",
            energy_pred_col=f"{energy_col}_pred",
        )
        path = out_dir / f"predictions_{name}.csv"
        df_pred.to_csv(path, index=False)
        print(f"Saved predictions: {path}")

    metrics_path = results_dir / "metrics_report.txt"
    metrics_path.write_text(
        format_metrics_txt(metrics_pop, metrics_ind, best_name, score_pop, score_ind),
        encoding="utf-8",
    )
    print(f"Saved metrics report: {metrics_path}")


if __name__ == "__main__":
    main()