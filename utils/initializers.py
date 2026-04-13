import pandas as pd
import random
import re
import os
from models.EV import EV
from models.SmartEV import SmartEV
from config import RESULT_DIR


def clear_old_results():
    """Remove all files in the result directory."""
    for file_name in os.listdir(RESULT_DIR):
        file_path = os.path.join(RESULT_DIR, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def extract_ev_info(ev_path):
    """Extract the EV name and battery capacity from the file path."""
    filename = os.path.basename(ev_path)
    match = re.match(r"(EV_\d+)_([\d.]+)\.csv", filename)
    if match:
        ev_name = match.group(1)
        battery_capacity = float(match.group(2))
        return ev_name, battery_capacity
    else:
        raise ValueError(f"Invalid file format: {filename}")


def initialize_first_row(first, battery_capacity):
    """Prepare the initial row of trip data with default values."""
    initial_soc = 0.5
    initial_ebattery = initial_soc * battery_capacity
    initial_ebattG = initial_ebattR = initial_ebattery / 2

    nan_cols = ['NextDestPred', 'Plug_out_pred']
    default_values = {
        'arrival_SoC': initial_soc,
        'Ebattery': initial_ebattery,
        'EbattG': initial_ebattG,
        'EbattR': initial_ebattR
    }

    row = first.to_frame().T[['datetime', 'state', 'consumption']]

    for col in EV.columns:
        if col not in row.columns:
            row[col] = None if col in nan_cols else default_values.get(col, 0)

    return row


def create_inputs(ev_path, smart_charging, public, oracle, pred_type):
    input_trips = pd.read_csv(ev_path, parse_dates=['datetime'])
    name, battery_capacity = extract_ev_info(ev_path)



    first_row = initialize_first_row(input_trips.iloc[0], battery_capacity)

    ev_class = SmartEV if smart_charging else EV
    ev = (ev_class(name, input_trips, battery_capacity, False, pred_type) if not oracle else
          ev_class(name, input_trips, battery_capacity, oracle=True)) \
        if smart_charging else (ev_class(name, input_trips, battery_capacity)
                                if public else ev_class(name, input_trips, battery_capacity, public=False))

    ev.trips = pd.concat([ev.trips, first_row], ignore_index=True)
    return ev
