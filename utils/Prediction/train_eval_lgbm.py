from __future__ import annotations

import os
import pandas as pd

from lgbm_module import SplitConfig, TrainConfig, train_three_models


def main() -> None:
    df = pd.read_csv("data/charging_sessions.csv")

    split_cfg = SplitConfig(
        test_start="2020-01-06",
        test_days=28,
        lookback_days=365,
        val_days=14,
    )
    cfg = TrainConfig(
        user_col="user_id",
        place_col="place",
        time_col="plug_in_datetime",
        target_next_dest="next_dest",
        target_next_CBS="next_CBS",
        target_connected_duration="connected_duration",
        optuna_trials=40,
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    out = train_three_models(df=df, split_cfg=split_cfg, cfg=cfg, save_path="models/lgbm.joblib")
    print(f"Bundle saved to: {out['bundle_path']}")

    with open(os.path.join("results", "lgbm_test_metrics.txt"), "w") as f:
        for k, v in out["metrics_test"].items():
            f.write(f"{k}: {v}\n")

    df_test = out["df_test"].copy()
    for target, values in out["preds_test"].items():
        s = pd.Series(values, index=df_test.index)
        df_test[f"{target}_pred"] = s.values

    preds_path = os.path.join("output", "lgbm_test_predictions.csv")
    df_test.to_csv(preds_path, index=False)
    print(f"Wrote test predictions to: {preds_path}")


if __name__ == "__main__":
    main()