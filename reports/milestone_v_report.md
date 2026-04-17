# Milestone IV-V Diagnostics, Learning, and Synthesis

## 1. Inference diagnostics

### Variational inference diagnostics

We implemented coordinate-ascent variational inference (CAVI) in
[src/exact_inference.py](c:\Users\Administrateur\OneDrive - Al Akhawayn University in Ifrane\Documents\inferential\malaria_inference_project\src\exact_inference.py)
with explicit ELBO tracking.

Results on the 1000-observation synthetic malaria dataset:

- VI converged: `True`
- VI iterations to convergence: `60`
- Initial ELBO: `-12503.1471`
- Final ELBO: `-12478.9120`
- ELBO improvement: `24.2351`

Region-level posterior mean prevalence estimates under VI:

- Region 0: `0.5687`
- Region 1: `0.5322`
- Region 2: `0.5577`
- Region 3: `0.3051`
- Region 4: `0.8801`

To assess ELBO tightness, we computed an exact log evidence on a small subset
using latent-state enumeration in
[src/diagnostics.py](c:\Users\Administrateur\OneDrive - Al Akhawayn University in Ifrane\Documents\inferential\malaria_inference_project\src\diagnostics.py).

Small-subset VI tightness results:

- Subset size: `15` observations
- Final ELBO: `-191.1509`
- Exact log evidence: `-190.2903`
- ELBO gap: `0.8606`

Interpretation:

- The ELBO is fairly tight on the exact subset, with a gap below one nat.
- This indicates that the mean-field approximation is informative and not wildly underestimating posterior evidence.
- The gap is still nonzero, so the approximation is not exact; some posterior dependence is being ignored.

### MCMC diagnostics

We implemented a Gibbs sampler with multi-chain support in
[src/gibbs_sampler.py](c:\Users\Administrateur\OneDrive - Al Akhawayn University in Ifrane\Documents\inferential\malaria_inference_project\src\gibbs_sampler.py)
and diagnostics in
[src/diagnostics.py](c:\Users\Administrateur\OneDrive - Al Akhawayn University in Ifrane\Documents\inferential\malaria_inference_project\src\diagnostics.py).

Diagnostic results from 4 chains with 160 total draws per chain and 60 burn-in:

- ESS by region: `[78.12, 79.67, 55.73, 46.90, 51.97]`
- Split `R̂` by region: `[1.0560, 1.0253, 1.0821, 1.1311, 1.1196]`

Autocorrelation for Region 0 prevalence:

- Lag 1: `0.7431`
- Lag 2: `0.5967`
- Lag 3: `0.4632`
- Lag 4: `0.2490`
- Lag 5: `0.1373`
- Lag 6: `0.0635`

Interpretation:

- The chains are mixing, but autocorrelation is still substantial at short lags.
- ESS values are moderate rather than high.
- Regions 3 and 4 have `R̂ > 1.10`, which is a warning sign that the current MCMC budget is not fully sufficient for a strong convergence claim.

### VI fidelity against MCMC

We compared VI and MCMC posterior summaries directly:

- Mean absolute gap in individual posterior infection probabilities: `0.0157`
- Correlation between VI and MCMC individual posteriors: `0.9977`

Region-level mean gaps:

- Region 0: `0.0024`
- Region 1: `0.0107`
- Region 2: `0.0129`
- Region 3: `0.0027`
- Region 4: `0.0071`

Interpretation:

- Despite imperfect MCMC convergence, VI and MCMC agree closely on posterior means.
- The main practical discrepancy is not the central tendency, but posterior uncertainty and dependence structure.

### Reliability verdict

#### VI

The VI engine appears reliable for approximate posterior means on this dataset:

- It converges stably.
- Its ELBO improves monotonically in practice.
- Its posterior means match MCMC closely.
- Its ELBO gap on a tractable subset is small.

However, VI is still a mean-field approximation:

- It does not represent full posterior dependence.
- Its uncertainty may be overconfident relative to the exact posterior.

#### MCMC

The MCMC engine is usable but not fully converged under the current run budget:

- ESS is acceptable for exploratory work.
- Autocorrelation decays reasonably.
- But `R̂` above `1.10` for some regions means we should not yet claim strong convergence.

Bottom line:

- VI results are reliable enough for downstream learning and parameter estimation.
- MCMC results are directionally consistent, but stronger convergence would require longer chains.

## 2. Model-based learning extension

### Learning task

We framed learning as parameter estimation for the class-conditional Gaussian
emission model:

- Estimate `mu_0`, `sigma_0`, `mu_1`, `sigma_1`
- Use the inference engine as the E-step over latent infection states

The implementation is in
[src/learning.py](c:\Users\Administrateur\OneDrive - Al Akhawayn University in Ifrane\Documents\inferential\malaria_inference_project\src\learning.py)
via a variational EM routine:

- E-step: run CAVI to estimate posterior infection responsibilities
- M-step: update Gaussian means and variances using soft assignments

### Learning results

Original calibrated parameters:

- `mu_0 = [120.561, 5886.480]`
- `mu_1 = [109.893, 5162.380]`
- `sigma_0 = [11.696, 831.857]`
- `sigma_1 = [13.765, 1227.290]`

Learned parameters after 4 EM iterations:

- `mu_0 = [119.602, 5950.813]`
- `mu_1 = [107.920, 5143.962]`
- `sigma_0 = [11.621, 814.480]`
- `sigma_1 = [13.835, 1231.980]`

Learning ELBO:

- Final ELBO after learning: `-12472.4699`
- Baseline fixed-parameter VI ELBO: `-12478.9120`

Interpretation:

- The learned model improves the variational objective by about `6.44` nats.
- The recovered parameters remain close to the original calibrated values.
- This suggests the learning routine is recovering a sensible latent-class structure rather than drifting to degenerate solutions.

Region prevalence estimates after learning:

- Region 0: `0.5150`
- Region 1: `0.4874`
- Region 2: `0.5074`
- Region 3: `0.2744`
- Region 4: `0.8233`

Empirical prevalence from the synthetic labels:

- Region 0: `0.610`
- Region 1: `0.550`
- Region 2: `0.660`
- Region 3: `0.305`
- Region 4: `0.935`

Interpretation:

- The learned model preserves ordering across regions.
- The learned prevalence estimates are more conservative than the empirical labels.
- This is expected because the model is uncertainty-aware and the Gaussian classes overlap.

## 3. Critical synthesis

### Bayesian vs frequentist paradigm

The Bayesian framing shaped the project in three major ways:

1. Model design:
   The model explicitly represents latent infection states and region-level uncertainty using a Beta-Bernoulli hierarchy.

2. Algorithm choice:
   Instead of estimating only point prevalences, the engine computes posterior distributions using CAVI and Gibbs sampling.

3. Interpretation:
   Results are expressed as posterior means, credible intervals, entropy, and ELBO-based fit rather than only MLEs and confidence intervals.

If we had stayed fully frequentist, the natural design would likely have been logistic regression or empirical prevalence estimation. That would be faster and simpler, but it would not provide the same latent-state uncertainty quantification or hierarchical borrowing across regions.

### Unifying role of information theory

Information theory appears throughout the pipeline:

- VI optimizes the ELBO, which is equivalent to minimizing KL divergence between the variational approximation and the true posterior.
- The entropy term in the ELBO captures uncertainty in latent infection assignments.
- The ELBO gap quantifies how much posterior structure the variational family fails to represent.
- MCMC autocorrelation and ESS quantify how much information is effectively contained in correlated chains.

This makes information theory the common language connecting exactness, approximation quality, uncertainty, and computation.

### Pipeline limitations

Model limitations:

- The observation model assumes conditionally independent Gaussian features.
- Only two handcrafted image summary features are used.
- Region-level structure is simple and does not include covariates, time, or spatial dependence.

Algorithmic limitations:

- The current VI routine is accurate but slow in pure Python.
- The current Gibbs sampler still needs longer runs for stronger convergence claims.
- Mean-field VI cannot capture full posterior dependence.

Data limitations:

- The dataset is synthetic, even though it is calibrated from real malaria images.
- Synthetic labels may not reflect full real-world noise, artifact variation, or epidemiological shift.

## 4. Innovation proposal

### Extension idea

A promising extension is to adapt this inference engine to amortized inference in a neural-symbolic clinical surveillance system.

Example use case:

- A neural encoder extracts richer cell-image representations.
- A symbolic epidemiological layer imposes structured priors over prevalence, geography, and intervention history.
- An amortized inference network predicts local variational parameters for each case in real time.

This would preserve uncertainty-aware inference while making deployment much faster than per-dataset iterative optimization.

## 5. One-page research proposal

### Title

Amortized Hierarchical Inference for Neural-Symbolic Malaria Surveillance

### Refined inferential question

Can a neural amortized variational family approximate the posterior over latent malaria infection states and regional prevalence while preserving the interpretability and uncertainty quantification of a hierarchical Bayesian model?

### Proposed methodology

1. Replace handcrafted features with embeddings from a convolutional encoder trained on malaria cell images.
2. Retain the hierarchical latent structure:
   infection state at the individual level, prevalence at the regional level.
3. Introduce an amortized inference network that outputs local variational parameters for latent infection probabilities and region-level posterior factors.
4. Train the encoder and amortized inference network jointly by maximizing a structured ELBO.
5. Compare against:
   fixed-feature VI,
   Gibbs sampling on smaller subsets,
   frequentist discriminative baselines.

### Evaluation plan

We would evaluate:

- Predictive performance:
  classification accuracy, ROC AUC, calibration
- Inferential fidelity:
  ELBO, KL-based diagnostics, agreement with MCMC on tractable subsets
- Uncertainty quality:
  credible interval coverage, entropy calibration, out-of-distribution sensitivity
- Efficiency:
  latency per batch and per patient, memory use, scalability with region count

### Expected contribution

The research would show whether amortization can preserve Bayesian interpretability while making inference fast enough for realistic AI-assisted surveillance pipelines.
