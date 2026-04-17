from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exact_inference import log_likelihood_gaussian
from .model_specification import MalariaModelConfig


@dataclass
class GibbsResult:
    pi_samples: np.ndarray
    z_samples: np.ndarray
    full_pi_trace: np.ndarray
    full_z_trace: np.ndarray
    regions: list[int]
    burn_in: int
    seed: int


class GibbsSampler:
    def __init__(self, df: pd.DataFrame, config: MalariaModelConfig):
        self.df = df.copy().reset_index(drop=True)
        self.config = config
        self.regions = sorted(self.df[config.region_column].unique())
        self.region_to_idx = {int(region): idx for idx, region in enumerate(self.regions)}
        self.region_values = self.df[config.region_column].to_numpy()
        self.feature_matrix = self.df.loc[:, list(config.feature_columns)].to_numpy(dtype=float)
        self.region_index = np.array([self.region_to_idx[int(region)] for region in self.region_values], dtype=int)
        self.log_like_z0 = np.array(
            [
                log_likelihood_gaussian(x, self.config.mu0, self.config.sigma0, self.config.mu1, self.config.sigma1, 0)
                for x in self.feature_matrix
            ],
            dtype=float,
        )
        self.log_like_z1 = np.array(
            [
                log_likelihood_gaussian(x, self.config.mu0, self.config.sigma0, self.config.mu1, self.config.sigma1, 1)
                for x in self.feature_matrix
            ],
            dtype=float,
        )

    def sample(
        self,
        n_samples: int = 2000,
        burn_in: int = 500,
        seed: int = 0,
        init_strategy: str = "prior",
    ) -> GibbsResult:
        if burn_in >= n_samples:
            raise ValueError("burn_in must be smaller than n_samples.")

        rng = np.random.default_rng(seed)
        n_obs = len(self.df)
        n_regions = len(self.regions)
        full_pi_trace = np.zeros((n_samples, n_regions), dtype=float)
        full_z_trace = np.zeros((n_samples, n_obs), dtype=int)

        if init_strategy == "prior":
            current_pi = rng.beta(self.config.alpha_hyper, self.config.beta_hyper, size=n_regions)
            current_z = rng.binomial(1, 0.5, size=n_obs)
        else:
            current_z = rng.binomial(1, 0.5, size=n_obs)
            current_pi = np.array(
                [
                    np.clip(current_z[self.region_values == region].mean(), 1e-3, 1.0 - 1e-3)
                    for region in self.regions
                ]
            )

        for draw in range(n_samples):
            for region_idx, region in enumerate(self.regions):
                region_mask = self.region_values == region
                region_z = current_z[region_mask]
                alpha_post = self.config.alpha_hyper + region_z.sum()
                beta_post = self.config.beta_hyper + region_mask.sum() - region_z.sum()
                current_pi[region_idx] = rng.beta(alpha_post, beta_post)

            pi_per_observation = np.clip(current_pi[self.region_index], 1e-8, 1.0 - 1e-8)
            log_p1 = np.log(pi_per_observation) + self.log_like_z1
            log_p0 = np.log(1.0 - pi_per_observation) + self.log_like_z0
            p1 = np.exp(log_p1 - np.logaddexp(log_p0, log_p1))
            current_z = rng.binomial(1, p1)

            full_pi_trace[draw] = current_pi
            full_z_trace[draw] = current_z

        return GibbsResult(
            pi_samples=full_pi_trace[burn_in:].copy(),
            z_samples=full_z_trace[burn_in:].copy(),
            full_pi_trace=full_pi_trace,
            full_z_trace=full_z_trace,
            regions=[int(region) for region in self.regions],
            burn_in=burn_in,
            seed=seed,
        )


def run_multiple_chains(
    df: pd.DataFrame,
    config: MalariaModelConfig,
    n_chains: int = 4,
    n_samples: int = 2000,
    burn_in: int = 500,
    seed: int = 11,
) -> list[GibbsResult]:
    sampler = GibbsSampler(df, config)
    results: list[GibbsResult] = []
    for chain_idx in range(n_chains):
        results.append(
            sampler.sample(
                n_samples=n_samples,
                burn_in=burn_in,
                seed=seed + 997 * chain_idx,
                init_strategy="prior",
            )
        )
    return results
