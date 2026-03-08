"""
environment.py
Responsibility: Environment creation, wrappers, and preprocessing for Assetto Corsa.
"""
import yaml
import assetto_corsa_gym.AssettoCorsaEnv.assettoCorsa as assettoCorsa

def create_env(root_config_path='./config.yaml'):
    """
    Initializes and returns the Assetto Corsa Gym environment using the root config.
    """
    # Load your top-level project config
    with open(root_config_path, 'r') as file:
        project_config = yaml.safe_load(file)
        
    # Extract the path to the inner Assetto Corsa configuration
    ac_config_path = project_config['environment']['ac_config_path']
    
    # Load the specific Assetto Corsa Gym config required by the simulator
    with open(ac_config_path, 'r') as file:
        ac_config = yaml.safe_load(file)
        
    # Create and return the environment
    env = assettoCorsa.make_assetto_corsa_env(config_dict=ac_config)
    return env