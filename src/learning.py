from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exact_inference import VariationalResult, coordinate_ascent_vi
from .model_specification import MalariaModelConfig


@dataclass
class LearningResult:
    learned_config: MalariaModelConfig
    variational_result: VariationalResult
    parameter_history: list[dict[str, float]]


class VariationalEMLearner:
    def __init__(self, df: pd.DataFrame, config: MalariaModelConfig):
        self.df = df.copy().reset_index(drop=True)
        self.config = config

    def fit(
        self,
        max_iter: int = 30,
        tol: float = 1e-4,
        min_sigma: float = 1.0,
    ) -> LearningResult:
        config = self.config
        prev_elbo = -np.inf
        history: list[dict[str, float]] = []
        responsibilities = None

        for iteration in range(1, max_iter + 1):
            vi_result = coordinate_ascent_vi(
                self.df,
                config,
                max_iter=200,
                tol=1e-6,
                init_responsibilities=responsibilities,
            )
            responsibilities = vi_result.responsibilities
            x = self.df.loc[:, list(config.feature_columns)].to_numpy(dtype=float)
            w1 = responsibilities
            w0 = 1.0 - responsibilities

            mean0 = np.average(x, axis=0, weights=w0)
            mean1 = np.average(x, axis=0, weights=w1)
            var0 = np.average((x - mean0) ** 2, axis=0, weights=w0)
            var1 = np.average((x - mean1) ** 2, axis=0, weights=w1)
            sigma0 = np.sqrt(np.maximum(var0, min_sigma**2))
            sigma1 = np.sqrt(np.maximum(var1, min_sigma**2))

            config = MalariaModelConfig(
                alpha_hyper=config.alpha_hyper,
                beta_hyper=config.beta_hyper,
                mu_0=tuple(mean0.tolist()),
                sigma_0=tuple(sigma0.tolist()),
                mu_1=tuple(mean1.tolist()),
                sigma_1=tuple(sigma1.tolist()),
                feature_columns=config.feature_columns,
                region_column=config.region_column,
                latent_column=config.latent_column,
            )

            current_elbo = vi_result.elbo_history[-1]
            history.append(
                {
                    "iteration": float(iteration),
                    "elbo": float(current_elbo),
                    "mu0_feature_1": float(mean0[0]),
                    "mu1_feature_1": float(mean1[0]),
                    "sigma0_feature_1": float(sigma0[0]),
                    "sigma1_feature_1": float(sigma1[0]),
                }
            )

            if abs(current_elbo - prev_elbo) < tol:
                break
            prev_elbo = current_elbo

        final_vi = coordinate_ascent_vi(
            self.df,
            config,
            max_iter=300,
            tol=1e-7,
            init_responsibilities=responsibilities,
        )
        return LearningResult(
            learned_config=config,
            variational_result=final_vi,
            parameter_history=history,
        )
