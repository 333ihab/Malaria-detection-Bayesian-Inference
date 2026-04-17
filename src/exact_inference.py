from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import norm

from .model_specification import MalariaModelConfig


@dataclass
class VariationalResult:
    dataframe: pd.DataFrame
    responsibilities: np.ndarray
    region_posteriors: dict[int, tuple[float, float]]
    elbo_history: list[float]
    converged: bool
    n_iter: int


def log_likelihood_gaussian(
    x: np.ndarray,
    mu_0: np.ndarray,
    sigma_0: np.ndarray,
    mu_1: np.ndarray,
    sigma_1: np.ndarray,
    z: int,
) -> float:
    if z == 0:
        return float(np.sum(norm.logpdf(x, mu_0, sigma_0)))
    return float(np.sum(norm.logpdf(x, mu_1, sigma_1)))


def _safe_binary_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def compute_elbo(
    df: pd.DataFrame,
    responsibilities: np.ndarray,
    region_posteriors: dict[int, tuple[float, float]],
    config: MalariaModelConfig,
) -> float:
    feature_matrix = df.loc[:, list(config.feature_columns)].to_numpy(dtype=float)
    regions = df[config.region_column].to_numpy()
    alpha0 = config.alpha_hyper
    beta0 = config.beta_hyper

    ll_z0 = np.array(
        [
            log_likelihood_gaussian(x, config.mu0, config.sigma0, config.mu1, config.sigma1, 0)
            for x in feature_matrix
        ]
    )
    ll_z1 = np.array(
        [
            log_likelihood_gaussian(x, config.mu0, config.sigma0, config.mu1, config.sigma1, 1)
            for x in feature_matrix
        ]
    )

    elbo = 0.0
    for region, (alpha_q, beta_q) in region_posteriors.items():
        region_mask = regions == region
        phi = responsibilities[region_mask]
        e_log_pi = digamma(alpha_q) - digamma(alpha_q + beta_q)
        e_log_one_minus_pi = digamma(beta_q) - digamma(alpha_q + beta_q)

        elbo += (
            (alpha0 - 1.0) * e_log_pi
            + (beta0 - 1.0) * e_log_one_minus_pi
            - (gammaln(alpha0) + gammaln(beta0) - gammaln(alpha0 + beta0))
        )
        elbo += np.sum(phi * e_log_pi + (1.0 - phi) * e_log_one_minus_pi)
        elbo += -(
            (alpha_q - 1.0) * e_log_pi
            + (beta_q - 1.0) * e_log_one_minus_pi
            - (gammaln(alpha_q) + gammaln(beta_q) - gammaln(alpha_q + beta_q))
        )

    elbo += float(np.sum(responsibilities * ll_z1 + (1.0 - responsibilities) * ll_z0))
    elbo += float(np.sum(_safe_binary_entropy(responsibilities)))
    return float(elbo)


def coordinate_ascent_vi(
    df: pd.DataFrame,
    config: MalariaModelConfig,
    max_iter: int = 200,
    tol: float = 1e-6,
    init_responsibilities: np.ndarray | None = None,
) -> VariationalResult:
    work_df = df.copy().reset_index(drop=True)
    n_obs = len(work_df)
    responsibilities = (
        np.full(n_obs, 0.5, dtype=float)
        if init_responsibilities is None
        else np.clip(np.asarray(init_responsibilities, dtype=float).copy(), 1e-6, 1.0 - 1e-6)
    )

    regions = sorted(work_df[config.region_column].unique())
    feature_matrix = work_df.loc[:, list(config.feature_columns)].to_numpy(dtype=float)
    region_values = work_df[config.region_column].to_numpy()
    elbo_history: list[float] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        region_posteriors: dict[int, tuple[float, float]] = {}
        for region in regions:
            region_mask = region_values == region
            soft_count = responsibilities[region_mask].sum()
            n_region = int(region_mask.sum())
            region_posteriors[int(region)] = (
                config.alpha_hyper + soft_count,
                config.beta_hyper + n_region - soft_count,
            )

        previous = responsibilities.copy()
        for idx, region in enumerate(region_values):
            alpha_q, beta_q = region_posteriors[int(region)]
            e_log_pi = digamma(alpha_q) - digamma(alpha_q + beta_q)
            e_log_one_minus_pi = digamma(beta_q) - digamma(alpha_q + beta_q)
            x = feature_matrix[idx]
            log_q1 = e_log_pi + log_likelihood_gaussian(
                x, config.mu0, config.sigma0, config.mu1, config.sigma1, 1
            )
            log_q0 = e_log_one_minus_pi + log_likelihood_gaussian(
                x, config.mu0, config.sigma0, config.mu1, config.sigma1, 0
            )
            responsibilities[idx] = np.exp(log_q1 - logsumexp([log_q0, log_q1]))

        elbo = compute_elbo(work_df, responsibilities, region_posteriors, config)
        elbo_history.append(elbo)

        max_delta = float(np.max(np.abs(responsibilities - previous)))
        if max_delta < tol:
            converged = True
            break

    result_df = work_df.copy()
    result_df["z_post"] = responsibilities
    return VariationalResult(
        dataframe=result_df,
        responsibilities=responsibilities,
        region_posteriors=region_posteriors,
        elbo_history=elbo_history,
        converged=converged,
        n_iter=iteration,
    )
