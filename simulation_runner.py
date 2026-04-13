from utils.initializers import create_inputs
import pandas as pd
from utils.predictions import predict_ev_charging


def complete_missing_lines(last_trip_time, target_time, ev):
    while last_trip_time < target_time:
        last_trip_time += pd.Timedelta(minutes=15)
        new_row = ev.create_row(last_trip_time, None, ev.trips.iloc[-1]['state'])
        ev.trips = pd.concat([ev.trips, pd.DataFrame([new_row])], ignore_index=True)


def find_next_datetime(ev, current_state, index):
    """Find the next datetime when the state changes."""
    next_datetime = None
    for j in range(index + 1, len(ev.inputs)):
        future_row = ev.inputs.iloc[j]
        if future_row['state'] != current_state:
            next_datetime = future_row['datetime']
            break

    if pd.isna(next_datetime):
        next_datetime = ev.inputs.iloc[-1]['datetime'] + pd.Timedelta(minutes=15)

    return next_datetime


def find_nd_CBS(ev, index):
    """Find the next destination and the CBS"""
    next_dest = "home"  # if EOF, consider the EV is going home
    cbs = 0.0
    for j in range(index + 1, len(ev.inputs)):
        future_row = ev.inputs.iloc[j]
        if future_row['state'] == 'driving':
            cbs += future_row['consumption']
        elif future_row['state'] != 'driving':
            prev = ev.inputs.iloc[j - 1]
            if prev['state'] == 'driving':
                next_dest = future_row['state']
                break

    return next_dest, cbs


def handle_driving_state(ev, row):
    """Handle the driving state in the pipeline."""
    last_trip_time = ev.trips['datetime'].iloc[-1] if not ev.trips.empty else None
    target_time = row['datetime'] - pd.Timedelta(minutes=15)

    complete_missing_lines(last_trip_time, target_time, ev)

    ev.trips = ev.trips[ev.trips['datetime'] < row['datetime']]
    new_row = ev.create_row(row['datetime'], None, row['state'])
    new_row['consumption'] = row['consumption']
    new_row['arrival_SoC'] -= row['consumption'] / ev.battery_capacity
    ev.arrival_SoC = new_row['arrival_SoC']
    new_row['Ebattery'] -= row['consumption']

    # Adjust energy levels based on consumption
    if new_row['EbattG'] >= row['consumption']:
        new_row['EbattG'] -= row['consumption']
    elif new_row['Ebattery'] > 0:
        new_row['EbattR'] -= row['consumption'] - new_row['EbattG']
        new_row['EbattG'] = 0.0
    else:
        new_row['EbattR'] = -1
        new_row['EbattG'] = -1
        print(f"Error: {ev.name} is out of battery!")
        ev.trips = pd.concat([ev.trips, pd.DataFrame([new_row])], ignore_index=True)
        return True  # Stop processing when out of battery

    ev.trips = pd.concat([ev.trips, pd.DataFrame([new_row])], ignore_index=True)

    if new_row['arrival_SoC'] <= 0.2:
        print(f"arrival_SoC is below 20% at {new_row['datetime']} for {ev.name}!")

    return False  # Continue processing


def handle_non_driving_state(ev, row):
    """Handle non-driving states (charging)."""
    if ev.smart and not ev.oracle:
        plug_out_time_pred, energy_needed_pred, next_destination_pred = predict_ev_charging(
    pd.to_datetime(row['datetime']), row['state'], ev.name, ev.pred_type, row['arrival_SoC'])
        # TODO : for simulation without prediction of next_destination, use :
        # next_destination_pred, _ = find_nd_CBS(ev, row.name)
        plug_out_time_pred = min(plug_out_time_pred, pd.to_datetime("2021-12-31 23:45:00"))
        ev.smart_charging(row['state'], pd.to_datetime(row['datetime']), pd.to_datetime(plug_out_time_pred),
                          energy_needed_pred, next_destination_pred)
    elif ev.smart and ev.oracle:
        next_datetime = find_next_datetime(ev, row['state'], row.name)
        next_dest, cbs = find_nd_CBS(ev, row.name)
        ev.smart_charging(row['state'], pd.to_datetime(row['datetime']), pd.to_datetime(next_datetime), cbs, next_dest)
    else:
        next_datetime = find_next_datetime(ev, row['state'], row.name)
        ev.charge_EV(pd.to_datetime(row['datetime']), next_datetime, row['state'])


def pipeline(ev):
    skip = False
    for index, row in ev.inputs.iterrows():
        if skip:
            if row['state'] == 'driving':
                skip = False
            else:
                continue
        current_state = row['state']
        if current_state == 'driving':
            stop_processing = handle_driving_state(ev, row)
            if stop_processing:
                break
        else:
            handle_non_driving_state(ev, row)
            skip = True
    last_trip_time = ev.trips['datetime'].iloc[-1]
    target_time = ev.inputs['datetime'].iloc[-1]
    complete_missing_lines(last_trip_time, target_time, ev)


def run_simulations(ev_paths, smart, public, oracle=False, pred_type=None):
    label = "SM" if smart else "noSM"
    for path in ev_paths:
        ev = create_inputs(path, smart, public, oracle, pred_type)
        print(f"Simulating {ev.name} ({'Smart' if smart else 'Standard'})...")
        pipeline(ev)
        if smart:
            trip_path = f"data_ev/trips_data/{ev.name}_{ev.battery_capacity}_{label}_{pred_type +'_' if pred_type is not None else ''}{'oracle_' if ev.oracle else ''}trips.csv"
            w_path = f"data_ev/workplace_data/{ev.name}_{ev.battery_capacity}_{label}_{pred_type +'_' if pred_type is not None else ''}{'oracle_' if ev.oracle else ''}.csv"
            h_path = f"data_ev/home_data/{ev.name}_{ev.battery_capacity}_{label}_{pred_type +'_' if pred_type is not None else ''}{'oracle_' if ev.oracle else ''}.csv"
        else:
            trip_path = f"data_ev/trips_data/{ev.name}_{ev.battery_capacity}_{label}_{'noPublic_' if not public else ''}trips.csv"
            w_path = f"data_ev/workplace_data/{ev.name}_{ev.battery_capacity}_{label}_{'noPublic_' if not public else ''}.csv"
            h_path = f"data_ev/home_data/{ev.name}_{ev.battery_capacity}_{label}_{'noPublic_' if not public else ''}.csv"
        ev.trips.iloc[1:].to_csv(trip_path, index=False)
        ev.to_home.to_csv(h_path, index=False)
        ev.to_workplace.to_csv(w_path, index=False)


def simulate_evs(ev_files, mode):
    """Run simulations based on the selected mode."""
    if mode in ("smart_gmm_i", "smart_gmm_p", "smart_lgbm", "smart_sims","smart_2step"):
        pred_type = mode[6:]
        run_simulations(ev_files, smart=True, public=True, pred_type=pred_type)

    if mode == "non_smart":
        run_simulations(ev_files, smart=False, public=True)

    if mode == "non_smart_no_public":
        run_simulations(ev_files, smart=False, public=False)

    if mode == "smart_oracle":
        run_simulations(ev_files, smart=True, public=True, oracle=True)
