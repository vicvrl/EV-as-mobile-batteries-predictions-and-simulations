import pandas as pd


def get_energy(REC_data, current_time):
    """Retrieve energy values from REC_data."""
    energy = REC_data.loc[REC_data['datetime'] == current_time]["energy"].values[0]
    return energy


def get_place_power(place):
    return {
        "home": 3.7,
        "workplace": 11,
        "public": 22,
        "fast75": 75,
        "fast150": 150
    }.get(place, None)


def get_REC_predictions(arrival_time, departure_time, current_place):
    """Simulates a communication with the REC.
    Get the predictions from the REC for the connected duration and the current place
    :param arrival_time: the time when the car is connected
    :param departure_time: the time when the car is disconnected
    :param current_place: the location where the car is connected
    :return: the predictions from the REC"""
    path = "data_REC/" + current_place + "/data.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df[(df["datetime"] >= arrival_time) & (df["datetime"] <= (departure_time + pd.Timedelta("15min")))]


def compute_Eavailable(REC_predictions):
    """
    Compute the available energy surplus from REC predictions.
    Supports both 'power' and 'energy' formats.

    :param REC_predictions: DataFrame with 'datetime' and either 'power' or 'energy' column.
    :return: tuple (surplus energy, timestamps with no surplus)
    """
    REC_predictions = REC_predictions.sort_values(by="datetime").copy()

    if "energy" in REC_predictions.columns:
        # Use the positive energy values directly
        valid = REC_predictions["energy"] > 0
        REC = REC_predictions.loc[valid, "energy"].sum()
        no_surplus_moments = REC_predictions.loc[~valid, "datetime"].values

    else:
        raise ValueError("REC_predictions must contain either an 'energy' column.")

    return REC, no_surplus_moments


def compute_Eneeded(Epredicted, EbattCap):
    """Compute the energy needed to reach the next destination
    :param EbattCap: the battery capacity
    :param Epredicted: the predicted energy needed
    :return: the energy needed"""

    anxiety = 1.5
    margin = Epredicted * anxiety
    Eunpredicted = 0.2 * EbattCap + margin
    return (Epredicted + Eunpredicted)/0.95

