import pandas as pd
import os


def compute_selfconsumption(ev_name, trips_path, smart_flag, output_path):
    df = pd.read_csv(trips_path)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    df['delta_EbattR'] = df['EbattR'].diff().fillna(0)
    df['delta_EbattR'] = df['delta_EbattR'].apply(lambda x: x if x > 0 else 0)

    results = []

    for month in range(1, 13):
        monthly_df = df[df['datetime'].dt.month == month]
        if monthly_df.empty:
            continue

        home_cons = monthly_df[monthly_df['state'] == 'home']['delta_EbattR'].sum()
        workplace_cons = monthly_df[monthly_df['state'] == 'workplace']['delta_EbattR'].sum()

        results.append({
            'ev_name': ev_name,
            'smart_charging': smart_flag,
            'month': month,
            'workplace': workplace_cons,
            'home': home_cons
        })

    result_df = pd.DataFrame(results)
    file_exists = os.path.exists(output_path)
    result_df.to_csv(output_path, mode='a', header=not file_exists, index=False)


def compute_from_grid(ev_name, home_path, workplace_path, smart_flag, output_path):
    df_home = pd.read_csv(home_path)
    df_workplace = pd.read_csv(workplace_path)

    df_home['datetime'] = pd.to_datetime(df_home['datetime'], errors='coerce')
    df_workplace['datetime'] = pd.to_datetime(df_workplace['datetime'], errors='coerce')

    df_home['total'] = df_home['energy'] + df_home['from_ev']
    df_workplace['total'] = df_workplace['energy'] + df_workplace['from_ev']

    results = []

    for month in range(1, 13):
        home_month_df = df_home[df_home['datetime'].dt.month == month]
        workplace_month_df = df_workplace[df_workplace['datetime'].dt.month == month]

        if home_month_df.empty and workplace_month_df.empty:
            continue

        grid_power_home = home_month_df[home_month_df['total'] < 0]['total'].sum()
        grid_power_workplace = workplace_month_df[workplace_month_df['total'] < 0]['total'].sum()

        ev_power_home = home_month_df[home_month_df['from_ev'] > 0]['from_ev'].sum()
        ev_power_workplace = workplace_month_df[workplace_month_df['from_ev'] > 0]['from_ev'].sum()

        results.append({
            'ev_name': ev_name,
            'smart_charging': smart_flag,
            'month': month,
            'grid_home': grid_power_home,
            'grid_workplace': grid_power_workplace,
            'ev_home': ev_power_home,
            'ev_workplace': ev_power_workplace
        })

    result_df = pd.DataFrame(results)
    file_exists = os.path.exists(output_path)
    result_df.to_csv(output_path, mode='a', header=not file_exists, index=False)


def find_next_datetime(df, current_state, index):
    """Find the next datetime when the state changes."""
    next_datetime = None
    i = 0
    for j in range(index + 1, len(df)):
        future_row = df.iloc[j]
        if future_row['state'] != current_state:
            next_datetime = future_row['datetime']
            i = j
            break

    if pd.isna(next_datetime):
        next_datetime = df.iloc[-1]['datetime'] + pd.Timedelta(minutes=15)
        i = len(df)

    return next_datetime, i


def timeseries_to_charging_sessions(folder):
    results = []
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    for file in files:
        ev_name = file.replace('.csv', '')
        print(f'Processing {ev_name}')
        df = pd.read_csv(f'{folder}/{file}', parse_dates=['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        j = 0
        while j < len(df):
            row = df.iloc[j]
            if row['state'] != 'driving':  # new charging session
                state = row['state']
                discharged_first = 0
                # create a new session
                plug_in_time = row['datetime']
                plug_out_time, i = find_next_datetime(df, row['state'], j)
                while df.iloc[j]['EchargedBattery'] == 0 and j < len(df) - 1 and df.iloc[j]['state'] != 'driving':
                    j += 1
                energy = df.iloc[j]['EchargedBattery']
                j += 1
                while j < len(df) - 1 and df.iloc[j]['EchargedBattery'] < 0 and df.iloc[j]['state'] != 'driving':
                    energy += df.iloc[j]['EchargedBattery']
                    j += 1
                if energy < 0:
                    discharged_first = 1
                    energy = -energy
                session = {
                    'ev_name': ev_name,
                    'plug_in_time': plug_in_time,
                    'plug_out_time': plug_out_time,
                    'state': state,
                    'discharged_first': discharged_first,
                    'energy': energy if energy < 0 else 0
                }
                j = i
                results.append(session)
            else:
                j += 1
    final_df = pd.DataFrame(results)
    final_df.to_csv('../results/all_ev_charging_sessions_with_discharge.csv', index=False)


# timeseries_to_charging_sessions('../data_ev/trips_data')


def compute_community_transfers(session_csv_path, output_path_csv='../results/community_transfers.csv'):
    df = pd.read_csv(session_csv_path, parse_dates=['plug_in_time', 'plug_out_time'])

    # Extract sim_type from ev_name like: EV_0_42_SM_oracle_trips -> SM_oracle_trips
    df['sim_type'] = df['ev_name'].str.replace(r'^EV_\d+_\d+_', '', regex=True)

    # Only sessions with discharged energy
    df = df[df['discharged_first'] == 1].copy()

    # Sort by EV and session start time
    df = df.sort_values(['ev_name', 'plug_in_time']).reset_index(drop=True)

    # Determine direction by checking next non-public state for same EV
    directions = []
    for i, row in df.iterrows():
        current_state = row['state']
        ev_name = row['ev_name']

        # Search forward for next non-public session from same EV
        j = i + 1
        next_state = None
        while j < len(df):
            if df.loc[j, 'ev_name'] != ev_name:
                break
            state_j = df.loc[j, 'state']
            if state_j != 'public':
                next_state = state_j
                break
            j += 1

        # Determine direction
        if current_state == 'home' and next_state == 'workplace':
            directions.append('home_to_workplace')
        elif current_state == 'workplace' and next_state == 'home':
            directions.append('workplace_to_home')
        else:
            directions.append('unknown')  # skip these later

    df['direction'] = directions

    # Keep only known directions
    df = df[df['direction'] != 'unknown']

    # Group and summarize discharged energy
    summary = (
        df.groupby(['sim_type', 'direction'])['discharged_energy_computed']
            .sum()
            .unstack(fill_value=0)
    )

    # Make kWh positive for readability
    summary /= 110
    summary.to_csv(output_path_csv)


# compute_community_transfers('../results/charging_sessions_with_energy.csv')
