import time
start=time.time()
print('before import', flush=True)
from src.model_specification import default_config, load_synthetic_data
print('after import', time.time()-start, flush=True)
