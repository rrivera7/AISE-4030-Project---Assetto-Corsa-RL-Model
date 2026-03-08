"""
environment.py
Responsibility: Environment creation, wrappers, and preprocessing for Assetto Corsa.
"""
import sys
import os
from omegaconf import OmegaConf

# Route directly to the inner gym folder, identical to how was demonstrated in the demo notebook from the Assetto Corsa Gym repository
sys.path.append(os.path.abspath('./assetto_corsa_gym'))
import AssettoCorsaEnv.assettoCorsa as assettoCorsa

def create_env(config_path='./config.yaml'):
    """
    Initializes and returns the Assetto Corsa Gym environment using OmegaConf.
    """
    # Use OmegaConf to load the config exactly like the demo
    config = OmegaConf.load(config_path)
    
    # Create and return the environment using the specific make_ac_env function
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=config.work_dir)
    return env