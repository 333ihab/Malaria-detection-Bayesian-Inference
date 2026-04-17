from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import beta as beta_dist

from .exact_inference import VariationalResult, coordinate_ascent_vi, log_likelihood_gaussian
from .gibbs_sampler import GibbsResult
from .model_specification import MalariaModelConfig


@dataclass
class MCMCDiagnostics:
    ess: np.ndarray
    rhat: np.ndarray
    autocorrelation: dict[int, np.ndarray]


def autocorrelation(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    series = np.asarray(x, dtype=float)
    series = series - series.mean()
    variance = series.var()
    if variance <= 0:
        return np.ones(1, dtype=float)
    corr = np.correlate(series, series, mode="full")[len(series) - 1 :] / (variance * len(series))
    max_lag = min(max_lag, len(corr) - 1)
    return corr[: max_lag + 1]


def effective_sample_size(chains: np.ndarray, max_lag: int = 100) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws, n_params = chains.shape
    ess = np.zeros(n_params, dtype=float)
    for param_idx in range(n_params):
        acf_mean = np.mean(
            [autocorrelation(chains[c, :, param_idx], max_lag=max_lag) for c in range(n_chains)],
            axis=0,
        )
        positive_pairs = []
        for lag in range(1, len(acf_mean), 2):
            if lag + 1 >= len(acf_mean):
                break
            pair_sum = acf_mean[lag] + acf_mean[lag + 1]
            if pair_sum < 0:
                break
            positive_pairs.append(pair_sum)
        tau = 1.0 + 2.0 * np.sum(positive_pairs)
        ess[param_idx] = n_chains * n_draws / max(tau, 1e-8)
    return ess


def split_rhat(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws, n_params = chains.shape
    half = n_draws // 2
    if half < 2:
        raise ValueError("Need at least 4 post-burn-in draws per chain for split R-hat.")
    split = np.concatenate([chains[:, :half, :], chains[:, -half:, :]], axis=0)
    m = split.shape[0]
    n = split.shape[1]
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    between = n * chain_means.var(axis=0, ddof=1)
    within = chain_vars.mean(axis=0)
    var_hat = ((n - 1) / n) * within + between / n
    return np.sqrt(var_hat / within)


def summarize_mcmc(results: list[GibbsResult], max_lag: int = 60) -> MCMCDiagnostics:
    chains = np.stack([result.pi_samples for result in results], axis=0)
    ess = effective_sample_size(chains, max_lag=max_lag)
    rhat = split_rhat(chains)
    acf = {idx: autocorrelation(chains[0, :, idx], max_lag=max_lag) for idx in range(chains.shape[-1])}
    return MCMCDiagnostics(ess=ess, rhat=rhat, autocorrelation=acf)


def fit_beta_from_samples(samples: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(samples, dtype=float), 1e-6, 1.0 - 1e-6)
    mean = clipped.mean()
    var = clipped.var(ddof=1)
    concentration = max(mean * (1.0 - mean) / max(var, 1e-8) - 1.0, 2.0)
    alpha = max(mean * concentration, 1e-3)
    beta = max((1.0 - mean) * concentration, 1e-3)
    return alpha, beta


def beta_kl_divergence(alpha_p: float, beta_p: float, alpha_q: float, beta_q: float) -> float:
    grid = np.linspace(1e-4, 1.0 - 1e-4, 4000)
    p = beta_dist.pdf(grid, alpha_p, beta_p)
    q = beta_dist.pdf(grid, alpha_q, beta_q)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float(np.trapz(p * (np.log(p) - np.log(q)), grid))


def compare_vi_to_mcmc(
    vi_result: VariationalResult,
    mcmc_results: list[GibbsResult],
) -> dict[str, object]:
    chains = np.stack([result.pi_samples for result in mcmc_results], axis=0)
    pooled_pi = chains.reshape(-1, chains.shape[-1])
    vi_region_summary = {}
    for region_idx, region in enumerate(mcmc_results[0].regions):
        alpha_vi, beta_vi = vi_result.region_posteriors[region]
        alpha_mc, beta_mc = fit_beta_from_samples(pooled_pi[:, region_idx])
        vi_region_summary[region] = {
            "vi_mean": alpha_vi / (alpha_vi + beta_vi),
            "mcmc_mean": pooled_pi[:, region_idx].mean(),
            "mean_gap": abs(alpha_vi / (alpha_vi + beta_vi) - pooled_pi[:, region_idx].mean()),
            "kl_vi_to_mcmc_beta": beta_kl_divergence(alpha_vi, beta_vi, alpha_mc, beta_mc),
        }

    pooled_z = np.mean(np.stack([result.z_samples for result in mcmc_results], axis=0), axis=(0, 1))
    vi_probs = vi_result.responsibilities
    return {
        "region_summary": vi_region_summary,
        "z_posterior_correlation": float(np.corrcoef(vi_probs, pooled_z)[0, 1]),
        "z_mean_absolute_gap": float(np.mean(np.abs(vi_probs - pooled_z))),
    }


def exact_log_evidence_small_subset(df: pd.DataFrame, config: MalariaModelConfig) -> float:
    evidence = 0.0
    for region, region_df in df.groupby(config.region_column):
        features = region_df.loc[:, list(config.feature_columns)].to_numpy(dtype=float)
        n_region = len(features)
        if n_region > 14:
            raise ValueError("Exact evidence enumeration is only intended for small subsets (<=14 per region).")

        log_terms = []
        for state in product([0, 1], repeat=n_region):
            z = np.asarray(state, dtype=int)
            k = int(z.sum())
            log_prior = betaln(config.alpha_hyper + k, config.beta_hyper + n_region - k) - betaln(
                config.alpha_hyper, config.beta_hyper
            )
            ll = 0.0
            for obs_idx, obs in enumerate(features):
                ll += log_likelihood_gaussian(
                    obs, config.mu0, config.sigma0, config.mu1, config.sigma1, int(z[obs_idx])
                )
            log_terms.append(log_prior + ll)
        region_log_evidence = np.logaddexp.reduce(np.asarray(log_terms, dtype=float))
        evidence += float(region_log_evidence)
    return float(evidence)


def assess_vi_tightness(
    df: pd.DataFrame,
    config: MalariaModelConfig,
    subset_per_region: int = 8,
) -> dict[str, float]:
    subset = (
        df.groupby(config.region_column, group_keys=False)
        .head(subset_per_region)
        .reset_index(drop=True)
    )
    vi_result = coordinate_ascent_vi(subset, config, max_iter=300, tol=1e-8)
    exact_log_evidence = exact_log_evidence_small_subset(subset, config)
    final_elbo = vi_result.elbo_history[-1]
    return {
        "n_obs": float(len(subset)),
        "final_elbo": float(final_elbo),
        "exact_log_evidence": float(exact_log_evidence),
        "elbo_gap": float(exact_log_evidence - final_elbo),
    }
