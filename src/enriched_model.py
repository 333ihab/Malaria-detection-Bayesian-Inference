"""
Enriched Bayesian hierarchical model for malaria inference.

Upgrades over the baseline (MalariaModelConfig + coordinate_ascent_vi):

  1. Full covariance Gaussian (replaces diagonal diag(sigma^2))
  2. Mixture-of-Gaussians likelihood per class  (K components; K=1 -> single Gaussian)
  3. Region-specific feature offsets  (region -> X interaction)
  4. Label noise model  (Y_i | Z_i ~ Bernoulli(rho_{Z_i}))
  5. Learnable Beta hyperpriors  (alpha, beta updated via M-step with Gamma prior)

Inference: Variational EM
  E-step  -- update q(Z) = Bernoulli(phi_i) and q(pi_r) = Beta(alpha_r, beta_r) via CAVI
  M-step  -- update theta = {mix_weights, mu, cov, region_offsets, alpha, beta}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import multivariate_normal

from .model_specification import MalariaModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EnrichedConfig:
    """
    Configuration for the enriched Bayesian hierarchical model.

    Parameters
    ----------
    n_components : int
        Number of Gaussian mixture components per class (K).
        K=1 reduces to a single full-covariance Gaussian.
    mix_weights : ndarray (2, K)
        Mixture weights per class.
    mu : ndarray (2, K, D)
        Component means per class.
    cov : ndarray (2, K, D, D)
        Full covariance matrices per class/component.
    use_region_offsets : bool
        If True, learn a region-specific additive shift delta_r in feature space.
    region_offsets : ndarray (n_regions, D) or None
        Region-level feature offsets (learned in M-step).
    use_label_noise : bool
        If True, model observed label Y_i as noisy: Y_i | Z_i ~ Bernoulli(rho_{Z_i}).
    rho_sensitivity : float
        P(Y=1 | Z=1); probability of correctly detecting infection.
    rho_specificity : float
        P(Y=0 | Z=0); probability of correctly labelling healthy.
    learn_hyperpriors : bool
        If True, update alpha_hyper and beta_hyper in the M-step under
        a Gamma(gamma_shape, gamma_rate) hyperprior.
    """

    # Beta prior on regional prevalence
    alpha_hyper: float = 2.0
    beta_hyper: float = 2.0

    # MoG parameters
    n_components: int = 1
    mix_weights: np.ndarray = field(default=None, repr=False)  # (2, K)
    mu: np.ndarray = field(default=None, repr=False)            # (2, K, D)
    cov: np.ndarray = field(default=None, repr=False)           # (2, K, D, D)

    # Region -> feature interaction
    use_region_offsets: bool = False
    region_offsets: np.ndarray | None = field(default=None, repr=False)  # (R, D)

    # Label noise
    use_label_noise: bool = False
    rho_sensitivity: float = 0.95
    rho_specificity: float = 0.95
    observed_label_column: str | None = None

    # Learnable hyperpriors
    learn_hyperpriors: bool = False
    gamma_shape: float = 2.0
    gamma_rate: float = 1.0

    # Column names
    feature_columns: tuple[str, ...] = ("feature_1", "feature_2")
    region_column: str = "region"
    latent_column: str = "infection_latent"

    @classmethod
    def from_baseline(
        cls,
        config: MalariaModelConfig,
        n_components: int = 1,
        use_region_offsets: bool = False,
        use_label_noise: bool = False,
        learn_hyperpriors: bool = False,
        rho_sensitivity: float = 0.95,
        rho_specificity: float = 0.95,
        n_regions: int = 5,
        seed: int = 0,
    ) -> EnrichedConfig:
        """Build EnrichedConfig from a baseline MalariaModelConfig."""
        D = len(config.feature_columns)
        K = n_components
        rng = np.random.default_rng(seed)

        mu = np.zeros((2, K, D))
        cov = np.zeros((2, K, D, D))
        mix_weights = np.ones((2, K)) / K

        for k in range(K):
            # Slight perturbation so components start at different positions
            noise = rng.normal(0.0, 0.03, D)
            mu[0, k] = config.mu0 * (1.0 + noise)
            mu[1, k] = config.mu1 * (1.0 + noise)
            cov[0, k] = np.diag(config.sigma0 ** 2)
            cov[1, k] = np.diag(config.sigma1 ** 2)

        region_offsets = np.zeros((n_regions, D)) if use_region_offsets else None

        return cls(
            alpha_hyper=config.alpha_hyper,
            beta_hyper=config.beta_hyper,
            n_components=K,
            mix_weights=mix_weights,
            mu=mu.copy(),
            cov=cov.copy(),
            use_region_offsets=use_region_offsets,
            region_offsets=region_offsets,
            use_label_noise=use_label_noise,
            rho_sensitivity=rho_sensitivity,
            rho_specificity=rho_specificity,
            learn_hyperpriors=learn_hyperpriors,
            feature_columns=config.feature_columns,
            region_column=config.region_column,
            latent_column=config.latent_column,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EnrichedResult:
    dataframe: pd.DataFrame
    responsibilities: np.ndarray            # (n,)  phi_i = q(Z_i=1)
    region_posteriors: dict[int, tuple[float, float]]
    elbo_history: list[float]
    converged: bool
    n_iter: int
    final_config: EnrichedConfig


# ──────────────────────────────────────────────────────────────────────────────
# Likelihood helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_mog_batch(X: np.ndarray, config: EnrichedConfig, z: int) -> np.ndarray:
    """
    Vectorised log p(X | Z=z) under the mixture of Gaussians.
    Returns shape (n,).
    """
    K = config.n_components
    log_terms = np.empty((len(X), K))
    for k in range(K):
        log_terms[:, k] = np.log(max(config.mix_weights[z, k], 1e-12)) + multivariate_normal.logpdf(
            X, mean=config.mu[z, k], cov=config.cov[z, k], allow_singular=True
        )
    return logsumexp(log_terms, axis=1)  # (n,)


def _component_responsibilities(X: np.ndarray, config: EnrichedConfig, z: int) -> np.ndarray:
    """
    Soft component assignments gamma_{ik}^z = q(component=k | Z_i=z, X_i).
    Returns shape (n, K).
    """
    K = config.n_components
    log_gamma = np.empty((len(X), K))
    for k in range(K):
        log_gamma[:, k] = np.log(max(config.mix_weights[z, k], 1e-12)) + multivariate_normal.logpdf(
            X, mean=config.mu[z, k], cov=config.cov[z, k], allow_singular=True
        )
    return np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))  # (n, K)


# ──────────────────────────────────────────────────────────────────────────────
# M-step helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mstep_components(
    X: np.ndarray,
    phi: np.ndarray,
    config: EnrichedConfig,
    cov_reg: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update mixture weights, component means, and full covariance matrices.

    For class z=1 the effective weight of observation i for component k is:
        eff_{ik} = phi_i * gamma_{ik}^1
    For class z=0:
        eff_{ik} = (1-phi_i) * gamma_{ik}^0
    """
    D = X.shape[1]
    K = config.n_components
    mix_weights = config.mix_weights.copy()
    mu = config.mu.copy()
    cov = config.cov.copy()

    for z in range(2):
        z_weights = phi if z == 1 else (1.0 - phi)          # (n,)
        gamma = _component_responsibilities(X, config, z)    # (n, K)
        total_z = z_weights.sum() + 1e-12

        for k in range(K):
            eff = z_weights * gamma[:, k]                    # (n,)
            eff_sum = eff.sum() + 1e-12

            mix_weights[z, k] = eff_sum / total_z
            mu[z, k] = np.average(X, axis=0, weights=eff)

            diff = X - mu[z, k]                              # (n, D)
            # Weighted outer-product sum -> full covariance
            cov[z, k] = (eff[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / eff_sum
            cov[z, k] += cov_reg * np.eye(D)                 # regularise for invertibility

        # Renormalize mixture weights for this class
        mix_weights[z] = np.maximum(mix_weights[z], 1e-12)
        mix_weights[z] /= mix_weights[z].sum()

    return mix_weights, mu, cov


def _mstep_region_offsets(
    X: np.ndarray,
    phi: np.ndarray,
    region_index: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    """
    Update region offsets so that each region's soft feature mean (under Z=1)
    is centered on the global weighted mean.  delta_r captures the deviation.
    """
    D = X.shape[1]
    global_mean = np.average(X, axis=0, weights=phi + 1e-12)
    offsets = np.zeros((n_regions, D))
    for r in range(n_regions):
        mask = region_index == r
        if mask.sum() == 0:
            continue
        w = phi[mask] + 1e-12
        offsets[r] = np.average(X[mask], axis=0, weights=w) - global_mean
    return offsets


def _mstep_hyperpriors(
    region_posteriors: dict[int, tuple[float, float]],
    config: EnrichedConfig,
) -> tuple[float, float]:
    """
    Update alpha_hyper, beta_hyper by maximising:
        sum_r E_q[log Beta(pi_r; alpha, beta)]
        + log Gamma(alpha; gamma_shape, gamma_rate)
        + log Gamma(beta;  gamma_shape, gamma_rate)
    via L-BFGS-B in log-space (ensures positivity).
    """
    e_log_pi = np.array([
        digamma(aq) - digamma(aq + bq) for aq, bq in region_posteriors.values()
    ])
    e_log_1mpi = np.array([
        digamma(bq) - digamma(aq + bq) for aq, bq in region_posteriors.values()
    ])
    R = len(e_log_pi)
    a, b = config.gamma_shape, config.gamma_rate

    def neg_objective(log_params: np.ndarray) -> float:
        alpha, beta = np.exp(log_params)
        log_beta_norm = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        ll = R * (-log_beta_norm) + (alpha - 1.0) * e_log_pi.sum() + (beta - 1.0) * e_log_1mpi.sum()
        hyperprior = (a - 1.0) * np.log(alpha + 1e-12) - b * alpha \
                   + (a - 1.0) * np.log(beta  + 1e-12) - b * beta
        return -(ll + hyperprior)

    x0 = np.log([config.alpha_hyper, config.beta_hyper])
    result = minimize(neg_objective, x0, method="L-BFGS-B")
    alpha_new, beta_new = np.exp(result.x)
    return float(np.clip(alpha_new, 0.1, 100.0)), float(np.clip(beta_new, 0.1, 100.0))


# ──────────────────────────────────────────────────────────────────────────────
# ELBO
# ──────────────────────────────────────────────────────────────────────────────

def _compute_elbo(
    phi: np.ndarray,
    region_index: np.ndarray,
    regions: list[int],
    region_posteriors: dict[int, tuple[float, float]],
    ll_z0: np.ndarray,
    ll_z1: np.ndarray,
    config: EnrichedConfig,
    observed_labels: np.ndarray | None,
) -> float:
    alpha0, beta0 = config.alpha_hyper, config.beta_hyper
    elbo = 0.0

    for i_r, region in enumerate(regions):
        alpha_q, beta_q = region_posteriors[region]
        mask = region_index == i_r
        phi_r = phi[mask]
        e_log_pi   = digamma(alpha_q)                 - digamma(alpha_q + beta_q)
        e_log_1mpi = digamma(beta_q)                  - digamma(alpha_q + beta_q)

        # Prior: log p(pi_r | alpha0, beta0)
        elbo += (alpha0 - 1.0) * e_log_pi + (beta0 - 1.0) * e_log_1mpi
        elbo -= gammaln(alpha0) + gammaln(beta0) - gammaln(alpha0 + beta0)

        # E_q[log p(Z | pi_r)]
        elbo += float(np.sum(phi_r * e_log_pi + (1.0 - phi_r) * e_log_1mpi))

        # Entropy of q(pi_r) -- adds back - KL(q(pi_r) || prior) contribution
        elbo += gammaln(alpha_q) + gammaln(beta_q) - gammaln(alpha_q + beta_q)
        elbo -= (alpha_q - 1.0) * e_log_pi + (beta_q - 1.0) * e_log_1mpi

    # E_q[log p(X | Z)]
    elbo += float(np.sum(phi * ll_z1 + (1.0 - phi) * ll_z0))

    # Label noise: E_q[log p(Y | Z)]
    if config.use_label_noise and observed_labels is not None:
        rho1, rho0 = config.rho_sensitivity, config.rho_specificity
        y = observed_labels
        lp_y1 = y * np.log(rho1 + 1e-12) + (1.0 - y) * np.log(1.0 - rho1 + 1e-12)
        lp_y0 = y * np.log(1.0 - rho0 + 1e-12) + (1.0 - y) * np.log(rho0 + 1e-12)
        elbo += float(np.sum(phi * lp_y1 + (1.0 - phi) * lp_y0))

    # Entropy of q(Z)
    clipped = np.clip(phi, 1e-12, 1.0 - 1e-12)
    elbo += float(np.sum(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))))

    # Hyperprior contribution log p(alpha) + log p(beta)
    if config.learn_hyperpriors:
        a, b = config.gamma_shape, config.gamma_rate
        for val in (config.alpha_hyper, config.beta_hyper):
            elbo += (a - 1.0) * np.log(val + 1e-12) - b * val

    return float(elbo)


# ──────────────────────────────────────────────────────────────────────────────
# Main inference loop (Variational EM)
# ──────────────────────────────────────────────────────────────────────────────

def enriched_fit(
    df: pd.DataFrame,
    config: EnrichedConfig,
    max_outer: int = 50,
    max_inner: int = 30,
    tol: float = 1e-5,
    cov_reg: float = 1e-4,
) -> EnrichedResult:
    """
    Fit the enriched Bayesian hierarchical model via Variational EM.

    Each outer iteration runs:
      E-step  : CAVI updates for q(Z) and q(pi) with current theta fixed.
      M-step  : Closed-form / gradient updates for theta.

    Parameters
    ----------
    df        : DataFrame containing features and (optionally) labels.
    config    : EnrichedConfig specifying which upgrades are active.
    max_outer : Maximum number of Variational EM outer iterations.
    max_inner : Maximum CAVI steps per E-step.
    tol       : Convergence threshold on ELBO change (outer) and phi change (inner).
    cov_reg   : Diagonal regularisation added to covariance matrices.

    Returns
    -------
    EnrichedResult with fitted responsibilities, region posteriors, ELBO trace,
    and the final updated config.
    """
    work_df = df.copy().reset_index(drop=True)
    X_raw = work_df[list(config.feature_columns)].to_numpy(dtype=float)
    region_vals = work_df[config.region_column].to_numpy()
    regions = sorted(int(r) for r in np.unique(region_vals))
    region_to_idx = {r: i for i, r in enumerate(regions)}
    region_index = np.array([region_to_idx[int(r)] for r in region_vals], dtype=int)
    n = len(work_df)

    # Observed labels for label noise (fall back to latent column for synthetic data)
    observed_labels: np.ndarray | None = None
    if config.use_label_noise:
        col = config.observed_label_column or config.latent_column
        if col in work_df.columns:
            observed_labels = work_df[col].to_numpy(dtype=float)

    # Initialise variational parameters
    phi = np.full(n, 0.5)
    region_posteriors: dict[int, tuple[float, float]] = {
        r: (config.alpha_hyper, config.beta_hyper) for r in regions
    }
    elbo_history: list[float] = []
    converged = False

    for outer_iter in range(1, max_outer + 1):

        # ── Apply region offsets to raw features ─────────────────────────
        X = X_raw.copy()
        if config.use_region_offsets and config.region_offsets is not None:
            for i_r, region in enumerate(regions):
                mask = region_index == i_r
                X[mask] -= config.region_offsets[i_r]   # subtract offset -> center features

        # ── Pre-compute log-likelihoods (fixed theta during E-step) ──────
        ll_z0 = _log_mog_batch(X, config, z=0)
        ll_z1 = _log_mog_batch(X, config, z=1)

        # ── E-step: CAVI inner loop ───────────────────────────────────────
        for _ in range(max_inner):
            # Update region posteriors (Beta conjugate update)
            for i_r, region in enumerate(regions):
                mask = region_index == i_r
                sc = phi[mask].sum()
                nr = int(mask.sum())
                region_posteriors[region] = (
                    config.alpha_hyper + sc,
                    config.beta_hyper + nr - sc,
                )

            # Vectorised responsibility update
            e_log_pi   = np.array([digamma(region_posteriors[regions[r]][0]) - digamma(sum(region_posteriors[regions[r]])) for r in region_index])
            e_log_1mpi = np.array([digamma(region_posteriors[regions[r]][1]) - digamma(sum(region_posteriors[regions[r]])) for r in region_index])

            log_q1 = e_log_pi   + ll_z1
            log_q0 = e_log_1mpi + ll_z0

            # Label noise adjustment
            if config.use_label_noise and observed_labels is not None:
                rho1, rho0 = config.rho_sensitivity, config.rho_specificity
                y = observed_labels
                log_q1 += y * np.log(rho1 + 1e-12) + (1.0 - y) * np.log(1.0 - rho1 + 1e-12)
                log_q0 += y * np.log(1.0 - rho0 + 1e-12) + (1.0 - y) * np.log(rho0 + 1e-12)

            prev_phi = phi.copy()
            phi = np.exp(log_q1 - np.logaddexp(log_q0, log_q1))

            if np.max(np.abs(phi - prev_phi)) < tol:
                break

        # ── M-step ───────────────────────────────────────────────────────
        # 1. MoG parameters (mix weights, means, full covariances)
        new_weights, new_mu, new_cov = _mstep_components(X, phi, config, cov_reg=cov_reg)
        config.mix_weights = new_weights
        config.mu = new_mu
        config.cov = new_cov

        # 2. Region-specific feature offsets
        if config.use_region_offsets:
            config.region_offsets = _mstep_region_offsets(X_raw, phi, region_index, len(regions))

        # 3. Hyperpriors on alpha, beta
        if config.learn_hyperpriors:
            config.alpha_hyper, config.beta_hyper = _mstep_hyperpriors(region_posteriors, config)

        # ── ELBO ─────────────────────────────────────────────────────────
        elbo = _compute_elbo(phi, region_index, regions, region_posteriors, ll_z0, ll_z1, config, observed_labels)
        elbo_history.append(elbo)

        if len(elbo_history) >= 2 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            converged = True
            break

    result_df = work_df.copy()
    result_df["z_post_enriched"] = phi

    return EnrichedResult(
        dataframe=result_df,
        responsibilities=phi,
        region_posteriors=region_posteriors,
        elbo_history=elbo_history,
        converged=converged,
        n_iter=outer_iter,
        final_config=config,
    )
