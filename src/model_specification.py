from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MalariaModelConfig:
    alpha_hyper: float = 2.0
    beta_hyper: float = 2.0
    mu_0: tuple[float, ...] = (120.561, 5886.48)
    sigma_0: tuple[float, ...] = (11.696, 831.857)
    mu_1: tuple[float, ...] = (109.893, 5162.38)
    sigma_1: tuple[float, ...] = (13.765, 1227.29)
    feature_columns: tuple[str, ...] = ("feature_1", "feature_2")
    region_column: str = "region"
    latent_column: str = "infection_latent"

    @property
    def mu0(self) -> np.ndarray:
        return np.asarray(self.mu_0, dtype=float)

    @property
    def sigma0(self) -> np.ndarray:
        return np.asarray(self.sigma_0, dtype=float)

    @property
    def mu1(self) -> np.ndarray:
        return np.asarray(self.mu_1, dtype=float)

    @property
    def sigma1(self) -> np.ndarray:
        return np.asarray(self.sigma_1, dtype=float)


def default_config() -> MalariaModelConfig:
    return MalariaModelConfig()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_synthetic_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is not None:
        data_path = Path(path)
        return pd.read_csv(data_path)

    candidates = [
        project_root() / "notebooks" / "synthetic_malaria_data.csv",
        project_root() / "synthetic_malaria_data.csv",
        project_root() / "data" / "synthetic_malaria_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError("Could not locate synthetic_malaria_data.csv in the project.")
