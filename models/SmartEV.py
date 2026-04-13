import math
from models.EV import EV
from utils.helpers import compute_Eneeded, get_REC_predictions, compute_Eavailable, get_energy, \
    get_place_power
import pandas as pd


class SmartEV(EV):
    def __init__(self, name, trips, battery_capacity, oracle=False, pred_type=None):
        super().__init__(name, trips, battery_capacity)
        self.smart = True
        self.oracle = oracle
        self.pred_type = pred_type

    def new_row_predictions(self, current_time, plug_out_time, place, next_destination, Eneeded):
        new_row = self.create_row(current_time, plug_out_time, place)
        new_row['NextDestPred'] = next_destination
        new_row['Eneeded'] = Eneeded
        return new_row

    def battery_is_full(self, current_time, plug_out_time, place, next_destination, Eneeded):
        if self.trips.iloc[-1]["arrival_SoC"] >= EV.arrival_SoChLimit:
            while current_time < plug_out_time:
                new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")
        if current_time == plug_out_time:
            new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
            self.update_trips(new_row)

    def smart_charging(self, place, arrival_time, departure_time, Epredicted, next_destination):
        Eneeded = compute_Eneeded(Epredicted, self.battery_capacity)
        if place in ["home", "workplace"]:
            place_power = get_place_power(place)
            rec_predictions = get_REC_predictions(arrival_time, departure_time, place)
            surplus, no_surplus_moments = compute_Eavailable(rec_predictions)
            if self.trips.iloc[-1]['Ebattery'] < Eneeded:
                if surplus > 0:
                    if surplus >= Eneeded:
                        # charge the car until arrival_SoC hLimit is reached
                        self.charge_EV_REC(arrival_time, departure_time, place_power, place, rec_predictions,
                                           next_destination, Eneeded)
                    else:
                        # charge the car when surplus is present and add what's needed when no surplus
                        self.charge_EV_REC_and_grid(arrival_time, departure_time, place_power, place, rec_predictions,
                                                    Eneeded, surplus, next_destination)
                else:
                    self.next_destination_check(arrival_time, departure_time, place_power, place, next_destination,
                                                Eneeded)
            else:
                self.charge_EV_REC(arrival_time, departure_time, place_power, place, rec_predictions, next_destination,
                                   Eneeded)
        elif place in ["public", "fast75", "fast150"]:
            self.next_destination_check(arrival_time, departure_time, get_place_power(place), place, next_destination,
                                        Eneeded)
        else:
            print("Invalid location type")
            return

    def next_destination_check(self, plug_in_time, plug_out_time, place_power, place, next_destination, Eneeded):
        if next_destination in ["home", "workplace"]:
            if self.arrival_SoC > Eneeded:
                current_time = plug_in_time
                while current_time < plug_out_time:
                    new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
                    self.update_trips(new_row)
            else:
                self.charge_EV_grid(plug_in_time, plug_out_time, place_power, place, Eneeded, next_destination)
        elif next_destination in ["public", "fast75", "fast150"]:
            self.charge_EV_grid(plug_in_time, plug_out_time, place_power, place, Eneeded, next_destination)

    def charge_EV_REC(self, plug_in_time, plug_out_time, place_power, place, REC_data, next_destination, Eneeded):
        current_time = plug_in_time
        while current_time < plug_out_time and self.trips.iloc[-1]["arrival_SoC"] < EV.arrival_SoChLimit:
            new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
            energy_value = get_energy(REC_data, current_time)
            if energy_value > 0:
                energy_place = place_power * 0.25
                if energy_value > energy_place:  # cannot charge more than the place power
                    energy_value = energy_place
                new_row = self.charge_from_REC(new_row, energy_value)
            else:
                new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
                new_row = self.discharge_to_REC(current_time, plug_out_time, Eneeded, place_power, place, new_row)
            self.update_trips(new_row)
            current_time += pd.Timedelta("15min")
        if self.trips.iloc[-1]["arrival_SoC"] >= EV.arrival_SoChLimit:
            while current_time < plug_out_time:
                new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
                new_row = self.discharge_to_REC(current_time, plug_out_time, Eneeded, place_power, place, new_row)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")
        if current_time == plug_out_time:
            new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
            self.update_trips(new_row)

    def charge_EV_grid(self, plug_in_time, plug_out_time, place_power, place, Eneeded, next_destination):
        current_time = plug_in_time
        while current_time < plug_out_time and self.trips.iloc[-1]["Ebattery"] < Eneeded:
            new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
            new_row = self.charge_from_grid(new_row, place_power * 0.25)
            self.update_trips(new_row)
            current_time += pd.Timedelta("15min")
        if self.trips.iloc[-1]["Ebattery"] >= Eneeded:
            while current_time < plug_out_time:
                new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
                self.update_trips(new_row)
                current_time += pd.Timedelta("15min")
        if current_time == plug_out_time:
            new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
            self.update_trips(new_row)

    def charge_EV_REC_and_grid(self, plug_in_time, plug_out_time, place_power, place, REC_data, Eneeded, surplus,
                               next_destination):
        from_grid_pred = Eneeded - surplus
        current_time = plug_in_time
        charged_from_grid = 0.0
        while current_time < plug_out_time and self.trips.iloc[-1]["arrival_SoC"] < EV.arrival_SoChLimit:
            new_row = self.new_row_predictions(current_time, plug_out_time, place, next_destination, Eneeded)
            energy_place = place_power * 0.25
            energy_value = get_energy(REC_data, current_time)
            if energy_value > energy_place:  # cannot charge more than the place power
                energy_value = energy_place
            if charged_from_grid >= from_grid_pred:
                new_row = self.charge_from_REC(new_row, energy_value if energy_value > 0 else 0)
            else:
                e_from_grid = energy_place - (energy_value if energy_value > 0 else 0)
                new_row = self.charge_from_REC(new_row, energy_value if energy_value > 0 else 0)
                self.charge_from_grid(new_row, e_from_grid)
                charged_from_grid += e_from_grid
            self.update_trips(new_row)
            current_time += pd.Timedelta("15min")
        self.battery_is_full(current_time, plug_out_time, place, next_destination, Eneeded)

    def discharge_to_REC(self, current_time, plug_out_time, Eneeded, power_place, place, new_row):
        last_row = self.trips.iloc[[-1]]

        dt = 0.25
        max_possible_energy = power_place * dt

        # Battery-based limits
        Ebattery_available = math.floor((last_row['Ebattery'].iloc[0] - Eneeded) * 10) / 10
        EbattR_available = math.floor(last_row['EbattR'].iloc[0] * 10) / 10

        # Actual dischargeable energy
        to_REC = min(max_possible_energy, Ebattery_available, EbattR_available) * EV.mu
        if to_REC > 0:
            if place == "home":
                self.to_home.loc[self.to_home['datetime'] == current_time, 'from_ev'] = to_REC
            elif place == 'workplace':
                self.to_workplace.loc[self.to_workplace['datetime'] == current_time, 'from_ev'] = to_REC

            new_row['EbattR'] -= to_REC
            new_row['Ebattery'] -= to_REC
            new_row['state'] = place
            new_row['Plug_out_pred'] = plug_out_time
            new_row['EchargedBattery'] = -to_REC
            new_row["Eneeded"] = Eneeded
            self.arrival_SoC -= to_REC / self.battery_capacity
            new_row['arrival_SoC'] = self.arrival_SoC
        else:
            new_row = self.charge_from_REC(new_row, 0)

        return new_row
