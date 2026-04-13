from __future__ import annotations

import os
import sys
import joblib

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, "Prediction"))

import pandas as pd

from lgbm_module import EVPredictionBundle
from sims_module import SimSConfig, SimilarSessionsModel
from gmm_modules import GMMDecisionBundle
from two_step_clustering_modules import (
    EVTwinStepBundle, SessionClusteringResult,
    PortfolioClusteringResult, PipelineConfig,
    _assign_pam, _assign_hier,
)

MODEL_DIR = os.path.join(_this_dir, "Prediction", "models")
DATA_PATH = os.path.join(_this_dir, "Prediction", "data", "charging_sessions.csv")

_lgbm_predictor     = None
_sims_predictor     = None
_gmm_predictor      = None
_two_step_predictor = None


def _get_lgbm_predictor() -> EVPredictionBundle:
    global _lgbm_predictor
    if _lgbm_predictor is None:
        _lgbm_predictor = EVPredictionBundle.load(os.path.join(MODEL_DIR, "lgbm.joblib"))
    return _lgbm_predictor


def _get_sims_predictor() -> SimilarSessionsModel:
    global _sims_predictor
    if _sims_predictor is None:
        df = pd.read_csv(DATA_PATH)
        _sims_predictor = SimilarSessionsModel(SimSConfig()).fit(df)
    return _sims_predictor


def _get_gmm_predictor() -> GMMDecisionBundle:
    global _gmm_predictor
    if _gmm_predictor is None:
        igmm_path = os.path.join(MODEL_DIR, "igmm_bundle.joblib")
        pgmm_path = os.path.join(MODEL_DIR, "pgmm_bundle.joblib")
        path = igmm_path if os.path.exists(igmm_path) else pgmm_path
        _gmm_predictor = GMMDecisionBundle.load(path)
    return _gmm_predictor


def _get_two_step_predictor() -> EVTwinStepBundle:
    global _two_step_predictor
    if _two_step_predictor is None:
        _mod = sys.modules["__main__"]
        _mod.EVTwinStepBundle          = EVTwinStepBundle
        _mod.SessionClusteringResult   = SessionClusteringResult
        _mod.PortfolioClusteringResult = PortfolioClusteringResult
        _mod.PipelineConfig            = PipelineConfig
        _mod._assign_pam               = _assign_pam 
        _mod._assign_hier              = _assign_hier
        _two_step_predictor = joblib.load(
            os.path.join(MODEL_DIR, "two_step_clustering.joblib")
        )
    return _two_step_predictor


def predict_ev_charging(datetime, place: str, user_id, pred_type: str, arrival_soc: float = 0.0):
    """
    Parameters
    ----------
    datetime    : plug-in datetime (str or pd.Timestamp)
    place       : categorical location string
    user_id     : user identifier (may be None for population GMM)
    pred_type   : 'lgbm' | 'sims' | 'gmm_p' | 'gmm_i' | '2step'
    arrival_soc : state of charge at plug-in (used by LGBM and 2step)

    Returns
    -------
    plug_out_time_exp : pd.Timestamp
    energy_needed_exp : float
    next_destination  : str
    """
    plug_in_dt = pd.to_datetime(datetime)
    pred_type  = (pred_type or "").strip().lower()

    if pred_type == "lgbm":
        bundle    = _get_lgbm_predictor()
        next_dest = bundle.predict_next_dest(user_id, place, plug_in_dt, arrival_soc=arrival_soc, top_k=1)[0]["next_dest"]
        next_CBS  = bundle.predict_next_CBS(user_id, place, plug_in_dt, arrival_soc=arrival_soc)
        duration  = bundle.predict_connected_duration(user_id, place, plug_in_dt, arrival_soc=arrival_soc)
        preds = {
            "pred_next_dest":          next_dest,
            "pred_next_CBS":           next_CBS,
            "pred_connected_duration": duration,
        }

    elif pred_type == "sims":
        predictor = _get_sims_predictor()
        cfg = predictor.cfg
        raw = predictor.predict_state(
            user_id=user_id,
            current_place=place,
            ts=plug_in_dt,
        )
        preds = {
            "pred_next_dest":          raw.get(cfg.dest_col),
            "pred_next_CBS":           raw.get(cfg.cons_col),
            "pred_connected_duration": raw.get(cfg.dur_col),
        }

    elif pred_type == "gmm_p":
        raw = _get_gmm_predictor().predict(
            place=place,
            arrival_time=plug_in_dt,
            user_id=None,
        )
        preds = {
            "pred_next_dest":          raw["destination_pred"],
            "pred_next_CBS":           raw["energy_pred"],
            "pred_connected_duration": raw["duration_pred"],
        }

    elif pred_type == "gmm_i":
        raw = _get_gmm_predictor().predict(
            place=place,
            arrival_time=plug_in_dt,
            user_id=user_id,
        )
        preds = {
            "pred_next_dest":          raw["destination_pred"],
            "pred_next_CBS":           raw["energy_pred"],
            "pred_connected_duration": raw["duration_pred"],
        }

    elif pred_type == "2step":
        result = _get_two_step_predictor().predict_single(
            user_id=user_id,
            place=place,
            plug_in_dt=plug_in_dt,
            arrival_soc=arrival_soc,
        )
        preds = {
            "pred_next_dest":          result["next_dest_pred"],
            "pred_next_CBS":           result["next_CBS_pred"],
            "pred_connected_duration": result["connected_duration_pred"],
        }

    else:
        raise ValueError(
            f"Unknown pred_type '{pred_type}'. "
            "Must be one of: 'lgbm', 'sims', 'gmm_p', 'gmm_i', '2step'."
        )

    connected_duration = float(preds["pred_connected_duration"])
    energy_needed_exp  = float(preds["pred_next_CBS"])
    next_destination   = preds["pred_next_dest"]

    plug_out_time_exp = (plug_in_dt + pd.Timedelta(hours=connected_duration)).round("min")

    return plug_out_time_exp, energy_needed_exp, next_destination