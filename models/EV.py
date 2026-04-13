from utils.helpers import get_REC_predictions, get_energy, get_place_power
import pandas as pd


class EV:
    mu = 0.95
    arrival_SoChLimit = 0.80

    columns = ["datetime", "state", "consumption", "arrival_SoC", "Ebattery", "EchargedBattery",
               "EbattR", "EbattG", "NextDestPred", "Plug_out_pred", "Eneeded"]

    def __init__(self, name, inputs, battery_capacity, public=True):
        self.name = name
        self.trips = pd.DataFrame(columns=EV.columns)
        self.inputs = inputs
        self.smart = False
        self.public = public
        self.battery_capacity = battery_capacity  # in kWh
        self.arrival_SoC = 0.5  # in kWh (initially 50%)
        self.to_workplace = pd.read_csv('data_REC/workplace/data.csv', parse_dates=['datetime']).copy()
        self.to_workplace['from_ev'] = 0.0
        self.to_home = pd.read_csv('data_REC/home/data.csv', parse_dates=['datetime']).copy()
        self.to_home['from_ev'] = 0.0

    def battery_is_full(self, current_time, plug_out_time, place, next_destination=None, Eneeded=None):
        if self.trips.iloc[-1]["arrival_SoC"] >= EV.arrival_SoChLimit:
            while current_time < plug_out_time:
                new_row = self.create_row(current_time, plug_out_time, place)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")
        if current_time == plug_out_time:
            new_row = self.create_row(current_time, plug_out_time, place)
            self.update_trips(new_row)

    def charge_EV(self, plug_in_time, plug_out_time, place):
        current_time = plug_in_time

        if place in ["home", "workplace"]:
            place_power = get_place_power(place)
            REC_data = get_REC_predictions(plug_in_time, plug_out_time, place)

            while current_time < plug_out_time and self.trips.iloc[-1]["arrival_SoC"] < EV.arrival_SoChLimit:
                new_row = self.create_row(current_time, plug_out_time, place)
                energy_value = get_energy(REC_data, current_time)
                new_row = self.apply_charging_strategy(new_row, energy_value, place_power)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")

        elif self.public:  # Public or fast charging
            while current_time < plug_out_time and self.trips.iloc[-1]["arrival_SoC"] < EV.arrival_SoChLimit:
                new_row = self.create_row(current_time, plug_out_time, place)
                place_power= get_place_power(place)
                new_row = self.charge_from_grid(new_row, place_power*0.25)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")
        else:
            # If not public activated, we assume no charging
            while current_time < plug_out_time:
                new_row = self.create_row(current_time, plug_out_time, place)
                new_row = self.charge_from_grid(new_row, 0)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")
        self.battery_is_full(current_time, plug_out_time, place)

    def apply_charging_strategy(self, new_row, energy_value, place_power):
        """Decides whether to charge from REC, Grid, or both."""
        place_energy = place_power * 0.25
        if energy_value >= place_energy:
            return self.charge_from_REC(new_row, place_energy)
        elif 0 < energy_value < place_energy:
            energy_from_grid = place_energy - energy_value
            new_row = self.charge_from_REC(new_row, energy_value)
            return self.charge_from_grid(new_row, energy_from_grid)
        else:
            return self.charge_from_grid(new_row, place_energy)

    def update_trips(self, new_row):
        self.trips = pd.concat([self.trips, pd.DataFrame([new_row])], ignore_index=True)

    def apply_energy(self, new_row, energy_value, source):
        E = energy_value * EV.mu
        new_row["Ebattery"] += E
        new_row["EchargedBattery"] += E
        new_row["arrival_SoC"] = new_row["Ebattery"] / self.battery_capacity
        self.arrival_SoC = new_row["arrival_SoC"]
        new_row[f"Ebatt{source}"] += E
        return new_row

    def charge_from_REC(self, new_row, energy_value):
        return self.apply_energy(new_row, energy_value, "R")

    def charge_from_grid(self, new_row, energy_value):
        return self.apply_energy(new_row, energy_value, "G")

    def create_row(self, time, plug_out_time, place):
        last = self.trips.iloc[-1]
        return {
            "datetime": time, "state": place, "consumption": 0.0,
            "arrival_SoC": last["arrival_SoC"], "Ebattery": last["Ebattery"],
            "EchargedBattery": 0.0, "EbattR": last["EbattR"],
            "EbattG": last["EbattG"], "NextDestPred": None,
            "Plug_out_pred": plug_out_time, "Eneeded": None
        }
