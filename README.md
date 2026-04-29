# Hierarchical Bayesian Malaria Infection Inference
CSC5341 – Probabilistic Modeling Project | Al Akhawayn University in Ifrane

---

## Project Overview

This project builds a complete **Bayesian hierarchical inference pipeline** for estimating malaria infection prevalence across geographic regions from cell-image features. The modelling problem mirrors a real clinical context: given microscopy image statistics from patient blood samples, infer (a) the latent infection status of each individual and (b) the regional prevalence of malaria, while propagating full posterior uncertainty rather than committing to point estimates.

---

## Probabilistic Model

### Hierarchical Generative Structure

$$\pi_r \sim \text{Beta}(\alpha, \beta), \quad r = 0, \ldots, 4$$

$$Z_i \mid \pi_{r(i)} \sim \text{Bernoulli}(\pi_{r(i)}), \quad i = 1, \ldots, n$$

$$\mathbf{X}_i \mid Z_i \sim \mathcal{N}(\boldsymbol{\mu}_{Z_i},\; \text{diag}(\boldsymbol{\sigma}_{Z_i}^2))$$

Where:
- $\pi_r$ is the region-level malaria prevalence
- $Z_i = 1$ indicates infection
- $\mathbf{X}_i \in \mathbb{R}^2$ contains microscopy-derived image features
- Feature distributions are calibrated to the [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

---

## Milestones

| # | Title | Key Contribution |
|---|-------|-----------------|
| I | Synthetic Data Generating Process | Kaggle-calibrated hierarchical DGP, 1,000-patient synthetic dataset |
| II | Coordinate-Ascent Variational Inference (CAVI) | Closed-form ELBO + mean-field approximate posterior |
| III | Gibbs Sampling | Vectorised multi-chain MCMC; asymptotically exact posterior |
| IV-C1 | Inference Diagnostics | ESS, split-$\hat{R}$, ELBO tightness, VI vs MCMC fidelity |
| IV-C2 | Variational EM Learning | Joint parameter learning; soft-assignment M-step |
| V | Critical Synthesis & Research Proposal | Information-theoretic analysis; amortised hierarchical VAE proposal |

---

## Key Results

| Engine | Status | Key Metric |
|--------|--------|------------|
| CAVI | Converged (60 iter) | ELBO gap = 1.789 nats (0.35%); 0.9997 correlation with MCMC |
| Gibbs (4 chains) | Fully converged | Min ESS = 431; Max $\hat{R}$ = 1.013 |
| Variational EM | 30 outer iterations | $\mu_0$ error = 1.51 units; $\mu_1$ error = 2.69 units |

---

## Project Structure

```
malaria_inference_project/
├── src/
│   ├── model_specification.py   # MalariaModelConfig, data loader
│   ├── exact_inference.py       # coordinate_ascent_vi(), compute_elbo()
│   ├── gibbs_sampler.py         # GibbsSampler (vectorised), run_multiple_chains()
│   ├── diagnostics.py           # ESS, R-hat, VI tightness, VI vs MCMC comparison
│   └── learning.py              # VariationalEMLearner (VI E-step + weighted MLE M-step)
├── notebooks/
│   ├── 01_synthetic_dgp.ipynb        # Milestone I
│   ├── 02_exact_inference.ipynb      # Milestone II
│   ├── 03_gibbs_sampler.ipynb        # Milestone III
│   ├── 04_diagnostics.ipynb          # Milestone IV-C1
│   ├── 05_model_extension.ipynb      # Milestone IV-C2
│   └── 06_synthesis.ipynb            # Milestone V
├── reports/
│   ├── full_project_report.md        # Complete technical report
│   └── milestone_v_report.md         # Milestone V standalone
├── synthetic_malaria_data.csv
└── README.md
```

---

## Research Proposal (Milestone V)

**Amortised Hierarchical Inference for Neural-Symbolic Malaria Surveillance**

The current pipeline is transductive — CAVI must be re-run per dataset. The proposed architecture replaces CAVI with a hierarchical VAE:

1. **CNN encoder** $f_\phi$ learns $q_\phi(Z_i \mid \mathbf{X}_i)$ directly from raw cell images.
2. **Structured posterior** preserves Beta-Bernoulli conjugacy for region-level aggregation.
3. **End-to-end training** via Gumbel-softmax reparameterisation.

Target: $>1000\times$ inference speedup over CAVI with equivalent posterior accuracy.

---

## Dependencies

```bash
pip install numpy pandas matplotlib scipy pillow
```

- Python 3.x
- numpy, pandas, matplotlib, scipy, pillow

---

## Author

**Ihab Kassimi**  
CSC5341 – Inferential Statistics, Spring 2026  
Al Akhawayn University in Ifrane
