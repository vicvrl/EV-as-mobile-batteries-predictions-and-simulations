from __future__ import annotations

import pandas as pd
from pathlib import Path

SOURCE_DIRS = [
    Path("../Dataset/EV_dataset/delivery_person/users_csv"),
    Path("../Dataset/EV_dataset/commuter/users_csv"),
    Path("../Dataset/EV_dataset/parents/users_csv"),
    Path("../Dataset/EV_dataset/remote_worker/users_csv"),
    Path("../Dataset/EV_dataset/unemployed/users_csv"),
]
OUTPUT_DIR = Path("data_EV/")

TEST_START = pd.Timestamp("2020-01-06")
TEST_END = TEST_START + pd.Timedelta(days=28)

KEEP_COLS = ["date", "state", "consumption"]
EMPTY_COLS = ["arrival_SoC", "Ebattery", "EchargedBattery", "EbattR", "EbattG", "NextDestPred", "Plug_out_pred", "Eneeded"]


def main() -> None:
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source_dir in SOURCE_DIRS:
        for csv_file in sorted(source_dir.glob("EV_*.csv")):
            parts = csv_file.stem.split("_")
            ev_id = parts[1]
            batt_size = parts[2]

            df = pd.read_csv(csv_file)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df_test = df[(df["date"] >= TEST_START) & (df["date"] < TEST_END)].copy()
            df_test = df_test.sort_values("date").reset_index(drop=True)

            if df_test.empty:
                print(f"[WARN] No test-period data for {csv_file.name}; skipping.")
                continue

            df_out = df_test[KEEP_COLS].rename(columns={"date": "datetime"}).copy()
            for col in EMPTY_COLS:
                df_out[col] = None
            df_out.at[0, "arrival_SoC"] = 0.5

            path = OUTPUT_DIR / f"EV_{ev_id}_{batt_size}.csv"
            df_out.to_csv(path, index=False)
            print(f"Saved {path.name} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()