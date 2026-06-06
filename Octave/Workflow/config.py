import yaml

def load_config(config_path: str) -> dict:
    """Load config"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config