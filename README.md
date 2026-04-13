# Electric Vehicles as Mobile Batteries: Predictions and Simulations

This repository contains the prediction models and simulation framework used to evaluate EVs as mobile batteries within Renewable Energy Communities (RECs) under a Community-to-Vehicle-to-Community (C2V2C) paradigm. It constitutes the second step of the pipeline, building on the synthetic EV dataset generated in the companion repository.

## Overview

This repository implements the full prediction and simulation pipeline for smart EV charging and discharging. Given a fleet of EV profiles (generated via the dataset generation repository), the framework:

- **Predicts** EV charging behavior: plug-out time, energy needed, and next destination
- **Simulates** both standard and smart charging/discharging strategies
- **Maximizes** REC self-consumption by aligning charging with local renewable energy surplus
- **Evaluates** performance across prediction methods and simulation modes

## Framework Architecture

```
main.py                          Entry point, CLI mode selection
config.py                        Paths and global settings
simulation_runner.py             Simulation pipeline per EV

models/
  EV.py                          Standard (non-smart) EV charging model
  SmartEV.py                     Smart EV model with C2V2C logic

utils/
  predictions.py                 Unified prediction interface
  helpers.py                     REC data access, energy calculations
  initializers.py                EV object creation and input loading
  metrics.py                     Self-consumption metric computation

utils/Prediction/
  lgbm_module.py                 LightGBM-based predictor
  gmm_modules.py                 Gaussian Mixture Model predictor
  sims_module.py                 Similar Sessions predictor
  two_step_clustering_modules.py Two-step clustering predictor and evaluation
  train_eval_lgbm.py             Training and evaluation for LightGBM
  train_eval_gmm.py              Training and evaluation for GMM
  evaluate_sims.py               Evaluation for similarity-based model

data_REC/
  home/data.csv                  REC energy data at home REC
  workplace/data.csv             REC energy data at workplace REC

data_ev/
  EV_*.csv                       Individual EV profiles (input)
  trips_data/                    Simulation output: trip-level records
  home_data/                     Simulation output: home charging records
  workplace_data/                Simulation output: workplace charging records
```

## Prediction Models

Three EV charging session attributes are predicted at each plug-in event:

| Target | Description |
|---|---|
| **Plug-out time** | When the EV will next depart |
| **Energy needed** | Consumption of the upcoming trip (kWh) |
| **Next destination** | Where the EV will drive to next |

Four prediction approaches are implemented and compared:

- **SM_TREE-LGBM (`lgbm`)** — Gradient boosting models with time-of-day cyclic encoding, trained per charging location. Uses Optuna for hyperparameter tuning.
- **SM_BEH-GMM-I (`gmm_i`)** — GMM fitted per EV, capturing individual driver behavior from historical sessions.
- **SM_BEH-GMM-P (`gmm_p`)** — GMM fitted on the full EV fleet, enabling prediction for new or data-sparse drivers.
- **SM_BEH-SIMS (`sims`)** — Retrieves the most similar past charging sessions based on arrival features and uses their outcomes as predictions.
- **SM_BEH-2Step-Clust (`2step`)** — First clusters sessions, then applies cluster-level distributions for prediction.

An **oracle mode** is also available, using ground-truth future values in place of predictions, providing an upper bound on achievable performance.

## Simulation Modes

The simulation loop processes each 15-minute timestep for every EV, applying the appropriate charging or discharging strategy based on the selected mode.

| Mode | Description |
|---|---|
| `non_smart` | Standard charging: charge as soon as plugged in, using REC surplus first |
| `non_smart_no_public` | Standard charging without public/fast charging |
| `smart_lgbm` | Smart charging using LightGBM predictions |
| `smart_gmm_i` | Smart charging using individual GMM predictions |
| `smart_gmm_p` | Smart charging using population GMM predictions |
| `smart_sims` | Smart charging using Similar Sessions predictions |
| `smart_2step` | Smart charging using Two-step Clustering predictions |
| `smart_oracle` | Smart charging using ground-truth future values (oracle) |

Smart charging decisions account for predicted plug-out time, energy needed, and next destination to schedule charging and discharging in alignment with REC renewable surplus.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running a simulation

```bash
python main.py --mode <mode>
```

**Example — smart charging with LightGBM predictions:**
```bash
python main.py --mode smart_lgbm
```

**Example — baseline non-smart charging:**
```bash
python main.py --mode non_smart
```

### Training prediction models

```bash
# Train and evaluate LightGBM model
python utils/Prediction/train_eval_lgbm.py

# Train and evaluate GMM model
python utils/Prediction/train_eval_gmm.py

# Evaluate similar sessions model
python utils/Prediction/evaluate_sims.py

# Train and evluate 2-step clustering model
python utils/Prediction/two_step_clustering_modules.py
```

## Input Data

Each EV profile is a CSV file placed in `data_ev/` and named `EV_<id>_<battery_capacity>.csv`. The file contains 15-minute resolution records with at minimum:

| Column | Description |
|---|---|
| `datetime` | Timestamp (15-min intervals) |
| `state` | EV state: `driving`, `home`, `workplace`, `public`, `fast75`, `fast150` |
| `consumption` | Energy consumed while driving (kWh) |
| `arrival_SoC` | State of Charge upon arrival (0–1) |

REC energy surplus data (used to determine available renewable energy at each charging point) is located in `data_REC/home/data.csv` and `data_REC/workplace/data.csv`.

## Output Data

For each EV and simulation mode, three output files are written:

| File | Location | Description |
|---|---|---|
| `<EV>_<cap>_<mode>_trips.csv` | `data_ev/trips_data/` | Full timestep-level simulation trace |
| `<EV>_<cap>_<mode>.csv` | `data_ev/home_data/` | Home charging records with REC contribution |
| `<EV>_<cap>_<mode>.csv` | `data_ev/workplace_data/` | Workplace charging records with REC contribution |

Output columns include `EbattR` (energy from renewables), `EbattG` (energy from grid), `EchargedBattery`, `NextDestPred`, `Plug_out_pred`, and `Eneeded`.

## Evaluation

Self-consumption metrics are computed per EV, per month, and per charging location using `utils/metrics.py`. Results are written to the `results/` directory.

## Citation

If you use this repository, please cite:

> Van Rillaer, V., de Schietere de Lophem, M., Verhaeghe, H. *Electric Vehicles as Mobile Batteries: Automating Charge and Discharge Using Machine Learning Predictions*

## Related Repository

The synthetic EV dataset used as input was generated using the companion repository:
**[EV_profiles_generation]([https://github.com/](https://github.com/vicvrl/EV_profiles_generation))** which generates 1,000 synthetic EV profiles at 15-minute resolution.
