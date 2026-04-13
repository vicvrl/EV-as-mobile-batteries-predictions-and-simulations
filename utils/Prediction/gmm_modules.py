from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


def arrival_to_sincos(arrival_ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(arrival_ts)
    minutes = ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60.0
    angle = 2.0 * np.pi * (minutes / 1440.0)
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _safe_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return mat / np.maximum(mat.sum(axis=1, keepdims=True), eps)


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    return amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))


@dataclass
class PlacePGMM:
    place: str
    n_components: int
    gmm: GaussianMixture
    scaler: StandardScaler
    dest_classes: List[str]
    phi: np.ndarray  # (K, C)
    feature_names: Tuple[str, ...] = ("arr_sin", "arr_cos", "duration", "next_CBS")

    @staticmethod
    def fit_from_df(
        df_place: pd.DataFrame,
        n_components_grid: List[int],
        n_init: int = 25,
        random_state: int = 42,
        cv_splits: int = 5,
        time_col: str = "plug_in_datetime",
        duration_col: str = "connected_duration",
        energy_col: str = "next_CBS",
        dest_col: str = "next_dest",
        verbose: bool = True,
    ) -> "PlacePGMM":
        if df_place.empty:
            raise ValueError("df_place is empty.")

        dfp = df_place.copy()
        dfp[time_col] = pd.to_datetime(dfp[time_col])
        dfp = dfp.sort_values(time_col).reset_index(drop=True)

        arr_sc = arrival_to_sincos(dfp[time_col])
        X = np.column_stack([
            arr_sc[:, 0], arr_sc[:, 1],
            dfp[duration_col].astype(float).to_numpy(),
            dfp[energy_col].astype(float).to_numpy(),
        ])

        dest_classes = sorted(dfp[dest_col].astype(str).unique().tolist())
        dest_to_idx = {c: i for i, c in enumerate(dest_classes)}
        y = dfp[dest_col].astype(str).map(dest_to_idx).to_numpy()

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, len(Xs) // 50)))

        def cv_score(K: int) -> float:
            scores = []
            for tr_idx, va_idx in tscv.split(Xs):
                gmm = GaussianMixture(
                    n_components=K, covariance_type="full",
                    n_init=n_init, random_state=random_state, reg_covar=1e-6,
                ).fit(Xs[tr_idx])
                scores.append(gmm.score(Xs[va_idx]))
            return float(np.mean(scores))

        best_k, best_score = None, -np.inf
        for K in n_components_grid:
            s = cv_score(K)
            if verbose:
                print(f"[{dfp['place'].iloc[0] if 'place' in dfp else 'place'}] K={K} CV loglik={s:.6f}")
            if s > best_score:
                best_score, best_k = s, K

        best_gmm = GaussianMixture(
            n_components=best_k, covariance_type="full",
            n_init=n_init, random_state=random_state, reg_covar=1e-6,
        ).fit(Xs)

        resp = best_gmm.predict_proba(Xs)
        C = len(dest_classes)
        phi = np.zeros((best_k, C), dtype=float)
        for k in range(best_k):
            w = resp[:, k]
            denom = w.sum() + 1e-12
            for c in range(C):
                phi[k, c] = np.sum(w * (y == c)) / denom
        phi = _safe_normalize_rows(phi)

        return PlacePGMM(
            place=str(dfp["place"].iloc[0]) if "place" in dfp.columns else "UNKNOWN",
            n_components=best_k,
            gmm=best_gmm,
            scaler=scaler,
            dest_classes=dest_classes,
            phi=phi,
        )

    def _arrival_scaled(self, arrival_time: pd.Timestamp) -> np.ndarray:
        arr = arrival_to_sincos(pd.Series([arrival_time]))
        return ((arr - self.scaler.mean_[:2]) / self.scaler.scale_[:2]).reshape(1, 2)

    def component_weights_given_arrival(
        self,
        arrival_time: pd.Timestamp,
        user_pi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        a = self._arrival_scaled(arrival_time)
        K = self.n_components
        pi = user_pi if user_pi is not None else self.gmm.weights_.copy()
        if user_pi is not None and user_pi.shape != (K,):
            raise ValueError(f"user_pi must have shape ({K},), got {user_pi.shape}")

        mu = self.gmm.means_[:, :2]
        cov = self.gmm.covariances_[:, :2, :2]
        x, d = a[0], 2

        log_probs = np.empty(K, dtype=float)
        for k in range(K):
            ck = cov[k] + 1e-9 * np.eye(2)
            sign, logdet = np.linalg.slogdet(ck)
            if sign <= 0:
                ck += 1e-6 * np.eye(2)
                sign, logdet = np.linalg.slogdet(ck)
            diff = x - mu[k]
            quad = float(diff.T @ np.linalg.inv(ck) @ diff)
            log_probs[k] = -0.5 * (quad + logdet + d * np.log(2 * np.pi))

        log_w = np.log(np.maximum(pi, 1e-12)) + log_probs
        return np.exp(log_w - _logsumexp(log_w.reshape(1, -1), axis=1)[0, 0])

    def predict_duration_energy(
        self,
        arrival_time: pd.Timestamp,
        user_pi: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        w = self.component_weights_given_arrival(arrival_time, user_pi=user_pi)
        a = self._arrival_scaled(arrival_time).reshape(2, 1)
        mu, cov, K = self.gmm.means_, self.gmm.covariances_, self.n_components

        cond_means_B = np.zeros((K, 2), dtype=float)
        for k in range(K):
            mu_A = mu[k, :2].reshape(2, 1)
            mu_B = mu[k, 2:].reshape(2, 1)
            Sigma_AA = cov[k, :2, :2] + 1e-9 * np.eye(2)
            cond = mu_B + cov[k, 2:, :2] @ np.linalg.inv(Sigma_AA) @ (a - mu_A)
            cond_means_B[k] = cond.reshape(-1)

        pred_B = (w.reshape(-1, 1) * cond_means_B).sum(axis=0)
        duration = max(float(pred_B[0] * self.scaler.scale_[2] + self.scaler.mean_[2]), 0.0)
        energy = max(float(pred_B[1] * self.scaler.scale_[3] + self.scaler.mean_[3]), 0.0)
        return duration, energy

    def predict_destination_proba(
        self,
        arrival_time: pd.Timestamp,
        user_pi: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        w = self.component_weights_given_arrival(arrival_time, user_pi=user_pi)
        proba = (w.reshape(-1, 1) * self.phi).sum(axis=0)
        proba /= proba.sum() + 1e-12
        return {cls: float(proba[i]) for i, cls in enumerate(self.dest_classes)}

    def predict_all(
        self,
        arrival_time: pd.Timestamp,
        user_pi: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        dest_proba = self.predict_destination_proba(arrival_time, user_pi=user_pi)
        duration, energy = self.predict_duration_energy(arrival_time, user_pi=user_pi)
        return {
            "destination_proba": dest_proba,
            "destination_pred": max(dest_proba, key=dest_proba.get),
            "duration_pred": duration,
            "energy_pred": energy,
        }


def personalize_weights_only(
    pgmm: PlacePGMM,
    df_user_place: pd.DataFrame,
    time_col: str = "plug_in_datetime",
    duration_col: str = "connected_duration",
    energy_col: str = "next_CBS",
    n_em_iters: int = 30,
    alpha_prior: float = 1.0,
) -> np.ndarray:
    if df_user_place.empty:
        raise ValueError("df_user_place is empty.")

    dfu = df_user_place.copy()
    dfu[time_col] = pd.to_datetime(dfu[time_col])
    arr_sc = arrival_to_sincos(dfu[time_col])
    X = np.column_stack([
        arr_sc[:, 0], arr_sc[:, 1],
        dfu[duration_col].astype(float).to_numpy(),
        dfu[energy_col].astype(float).to_numpy(),
    ])
    Xs = pgmm.scaler.transform(X)

    K, n, d = pgmm.n_components, Xs.shape[0], Xs.shape[1]
    mu, cov = pgmm.gmm.means_, pgmm.gmm.covariances_
    pi_u = pgmm.gmm.weights_.copy()

    log_pdf = np.zeros((n, K), dtype=float)
    for k in range(K):
        ck = cov[k] + 1e-9 * np.eye(d)
        sign, logdet = np.linalg.slogdet(ck)
        if sign <= 0:
            ck += 1e-6 * np.eye(d)
            sign, logdet = np.linalg.slogdet(ck)
        diff = Xs - mu[k]
        quad = np.sum(diff @ np.linalg.inv(ck) * diff, axis=1)
        log_pdf[:, k] = -0.5 * (quad + logdet + d * np.log(2 * np.pi))

    for _ in range(n_em_iters):
        log_r = np.log(np.maximum(pi_u, 1e-12)).reshape(1, -1) + log_pdf
        r = np.exp(log_r - _logsumexp(log_r, axis=1))
        pi_u = np.maximum(r.sum(axis=0) + (alpha_prior - 1.0), 1e-12)
        pi_u /= pi_u.sum()

    return pi_u


@dataclass
class GMMDecisionBundle:
    place_models: Dict[str, PlacePGMM]
    user_weights: Dict[str, Dict[str, np.ndarray]]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "GMMDecisionBundle":
        obj = joblib.load(path)
        if not isinstance(obj, GMMDecisionBundle):
            raise TypeError("Loaded object is not a GMMDecisionBundle.")
        return obj

    def get_user_pi(self, place: str, user_id: str) -> Optional[np.ndarray]:
        return self.user_weights.get(place, {}).get(str(user_id))

    def predict(
        self,
        place: str,
        arrival_time: pd.Timestamp,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if place not in self.place_models:
            raise KeyError(f"Unknown place '{place}'. Available: {list(self.place_models.keys())}")
        return self.place_models[place].predict_all(
            arrival_time=pd.to_datetime(arrival_time),
            user_pi=self.get_user_pi(place, str(user_id)) if user_id is not None else None,
        )