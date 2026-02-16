# Hierarchical Bayesian Malaria Infection Inference  
CSC5341 – Probabilistic Modeling Project  
Milestone I

---

## Project Overview

This project develops a hierarchical Bayesian model to estimate latent malaria infection states under regional prevalence uncertainty. The modeling framework is motivated by malaria surveillance challenges in Sub-Saharan Africa, where diagnostic uncertainty and heterogeneous regional prevalence directly affect treatment decisions and public health planning.

The primary objective is to construct a statistically realistic synthetic dataset with known ground-truth parameters. This enables rigorous validation of probabilistic inference algorithms in later milestones.

Specifically, the dataset is designed to:

- Capture region-level prevalence heterogeneity  
- Model latent infection states  
- Generate observable microscopy-derived features  
- Preserve known generative parameters for posterior recovery validation  

---

## Research Objective

We aim to infer:

- The latent infection state \( Z_{ir} \) of individual \( i \) in region \( r \)  
- The regional prevalence parameter \( \pi_r \)  
- Posterior uncertainty over infection probabilities  

This framework supports probabilistic diagnostic triage and regional prevalence monitoring under uncertainty.

Reference dataset for calibration:  
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

---

## Probabilistic Model

### Hierarchical Generative Structure

**1. Regional Prevalence**

\[
\pi_r \sim \text{Beta}(\alpha, \beta)
\]

Each region has its own prevalence parameter drawn from a shared Beta prior, introducing hierarchical structure.

**2. Latent Infection State**

\[
Z_{ir} \sim \text{Bernoulli}(\pi_r)
\]

Each individual’s infection state depends on their region’s prevalence.

**3. Observed Features**

\[
X_{ir} \mid Z=z \sim \mathcal{N}(\mu_z, \sigma_z^2)
\]

Conditional on infection status, observed microscopy-derived features follow class-specific Gaussian distributions.

Where:

- \( \mu_0, \sigma_0 \) correspond to healthy cells  
- \( \mu_1, \sigma_1 \) correspond to infected cells  
- Regional heterogeneity is induced through variation in \( \pi_r \)

---

## Kaggle-Based Calibration

Feature distributions are calibrated using the NIH Malaria Cell Images Dataset:

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

From 1000 images per class, we extract:

- Mean pixel intensity  
- Pixel intensity variance  

We estimate empirical values:

- \( \mu_0, \sigma_0 \) for uninfected cells  
- \( \mu_1, \sigma_1 \) for parasitized cells  

These statistics parameterize the Gaussian likelihood in the synthetic data-generating process (DGP), ensuring realistic feature scaling while maintaining controlled ground truth.

---

## Synthetic Data Generation

The synthetic dataset is generated as follows:

- Simulate 5 distinct regions  
- Generate 200 individuals per region  
- Sample region-level prevalence from a Beta distribution  
- Draw latent infection states from \( \text{Bernoulli}(\pi_r) \)  
- Generate Gaussian-distributed features conditional on infection status  

### Dataset Schema

| region | infection_latent | feature_1 | feature_2 |
|--------|------------------|-----------|-----------|

**Total observations:** 1000

---

## Validation

Empirical infection rates per region closely match their true generative parameters, with deviations attributable to sampling variability. This confirms correct implementation of the hierarchical Bernoulli process.

Feature distributions exhibit partial overlap between classes, motivating Bayesian posterior inference rather than deterministic classification thresholds.

---

## Project Structure

