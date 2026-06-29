import json

def load_configs(cfg_path: str, model_size: str):
    cfg = json.load(open(cfg_path))
    cfg = cfg[model_size]
    return cfg