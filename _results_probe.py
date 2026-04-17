import json
from src.model_specification import default_config, load_synthetic_data
from src.exact_inference import coordinate_ascent_vi
from src.gibbs_sampler import run_multiple_chains
from src.diagnostics import summarize_mcmc, compare_vi_to_mcmc, assess_vi_tightness
from src.learning import VariationalEMLearner

config = default_config()
df = load_synthetic_data()

vi = coordinate_ascent_vi(df, config, max_iter=150, tol=1e-6)
chains = run_multiple_chains(df, config, n_chains=4, n_samples=160, burn_in=60, seed=21)
mcmc_diag = summarize_mcmc(chains, max_lag=25)
vi_vs_mcmc = compare_vi_to_mcmc(vi, chains)
vi_tightness = assess_vi_tightness(df, config, subset_per_region=3)
learning = VariationalEMLearner(df, config).fit(max_iter=4, tol=1e-4)

payload = {
    'n_obs': len(df),
    'n_regions': int(df['region'].nunique()),
    'vi_converged': bool(vi.converged),
    'vi_iterations': int(vi.n_iter),
    'vi_initial_elbo': float(vi.elbo_history[0]),
    'vi_final_elbo': float(vi.elbo_history[-1]),
    'vi_region_means': {str(k): float(a / (a + b)) for k, (a, b) in vi.region_posteriors.items()},
    'mcmc_ess': [float(x) for x in mcmc_diag.ess],
    'mcmc_rhat': [float(x) for x in mcmc_diag.rhat],
    'acf_region0_first10': [float(x) for x in mcmc_diag.autocorrelation[0][:10]],
    'vi_vs_mcmc': vi_vs_mcmc,
    'vi_tightness': vi_tightness,
    'learned_mu0': [float(x) for x in learning.learned_config.mu_0],
    'learned_mu1': [float(x) for x in learning.learned_config.mu_1],
    'learned_sigma0': [float(x) for x in learning.learned_config.sigma_0],
    'learned_sigma1': [float(x) for x in learning.learned_config.sigma_1],
    'learning_final_elbo': float(learning.variational_result.elbo_history[-1]),
    'learning_iters': int(len(learning.parameter_history)),
    'learning_region_means': {str(k): float(a / (a + b)) for k, (a, b) in learning.variational_result.region_posteriors.items()},
}
print(json.dumps(payload, indent=2))
