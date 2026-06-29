import json

def load_configs(cfg_path: str):
    cfg = json.load(
        open(cfg_path)
    )

    return cfg