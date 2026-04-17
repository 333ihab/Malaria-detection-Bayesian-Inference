import time
from src.model_specification import default_config, load_synthetic_data
from src.exact_inference import coordinate_ascent_vi
from src.gibbs_sampler import run_multiple_chains
from src.diagnostics import summarize_mcmc, compare_vi_to_mcmc, assess_vi_tightness
from src.learning import VariationalEMLearner

config = default_config()
df = load_synthetic_data()

start = time.time()
vi = coordinate_ascent_vi(df, config, max_iter=150, tol=1e-6)
print('vi', round(time.time() - start, 3), flush=True)

start = time.time()
chains = run_multiple_chains(df, config, n_chains=4, n_samples=120, burn_in=40, seed=21)
print('mcmc', round(time.time() - start, 3), flush=True)

start = time.time()
diag = summarize_mcmc(chains, max_lag=20)
print('mcmc_diag', round(time.time() - start, 3), flush=True)

start = time.time()
cmp = compare_vi_to_mcmc(vi, chains)
print('vi_vs_mcmc', round(time.time() - start, 3), flush=True)

start = time.time()
tight = assess_vi_tightness(df, config, subset_per_region=3)
print('vi_tightness', round(time.time() - start, 3), flush=True)

start = time.time()
learning = VariationalEMLearner(df, config).fit(max_iter=4, tol=1e-4)
print('learning', round(time.time() - start, 3), flush=True)
