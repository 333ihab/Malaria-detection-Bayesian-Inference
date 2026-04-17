from src.model_specification import default_config, load_synthetic_data
print('imports ok', flush=True)
config = default_config()
df = load_synthetic_data()
print(df.shape, flush=True)
