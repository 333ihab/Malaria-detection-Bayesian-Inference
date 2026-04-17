import json
import numpy as np
from src.model_specification import default_config, load_synthetic_data
from src.exact_inference import coordinate_ascent_vi
from src.learning import VariationalEMLearner

config = default_config()
df = load_synthetic_data()
emp = df.groupby('region')['infection_latent'].mean().to_dict()
vi = coordinate_ascent_vi(df, config, max_iter=150, tol=1e-6)
learn = VariationalEMLearner(df, config).fit(max_iter=4, tol=1e-4)

payload = {
    'empirical_prevalence': {str(k): float(v) for k, v in emp.items()},
    'default_mu0': list(config.mu_0),
    'default_mu1': list(config.mu_1),
    'default_sigma0': list(config.sigma_0),
    'default_sigma1': list(config.sigma_1),
    'learned_mu0': list(learn.learned_config.mu_0),
    'learned_mu1': list(learn.learned_config.mu_1),
    'learned_sigma0': list(learn.learned_config.sigma_0),
    'learned_sigma1': list(learn.learned_config.sigma_1),
    'vi_region_means': {str(k): float(a/(a+b)) for k,(a,b) in vi.region_posteriors.items()},
    'learn_region_means': {str(k): float(a/(a+b)) for k,(a,b) in learn.variational_result.region_posteriors.items()},
}
print(json.dumps(payload, indent=2))
