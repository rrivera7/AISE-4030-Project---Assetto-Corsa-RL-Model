import sys
import os
import numpy as np
import gym
from omegaconf import OmegaConf
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath('./assetto_corsa_gym'))
import AssettoCorsaEnv.assettoCorsa as assettoCorsa


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Ensures the observation coming from AssettoCorsaEnv is a contiguous
    1-D float32 numpy array.  The underlying env already returns a flat
    Box space, so this wrapper only enforces the dtype contract that
    SB3's MlpPolicy expects.

    Args:
        env (gym.Env): The base Assetto Corsa Gym environment.
    """
    def __init__(self, env):
        # Initialize the base ObservationWrapper with the provided environment
        super().__init__(env)
        
        # Extract the shape of the original observation space
        obs_shape = env.observation_space.shape
        
        # Define a new observation space with the same shape but explicitly float32 dtype
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        """
        Casts the raw observation to float32.

        Args:
            obs (np.ndarray): Raw observation from the environment.

        Returns:
            np.ndarray: Observation guaranteed to be float32.
        """
        # Convert the incoming observation to a numpy array of type float32
        return np.asarray(obs, dtype=np.float32)


def create_env(config, log_dir='./models/sac_assetto_corsa/logs'):
    """
    Initializes and returns the wrapped Assetto Corsa Gym environment.
    Applies dtype normalization and SB3 Monitor for episode-level logging.

    Args:
        config (OmegaConf): The full project configuration (already loaded).
        log_dir (str): Directory where monitor.csv will be written.

    Returns:
        gym.Env: The ready-to-train environment instance.
    """
    # Retrieve the working directory from the configuration, defaulting to current directory
    work_dir = config.get('work_dir', './')

    # Create the base AC environment from the gym package using the provided config
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)

    # Wrap the environment to guarantee float32 observations for SB3 compatibility
    env = NormalizeObservationWrapper(env)

    # Ensure the logging directory exists, creating it if necessary
    os.makedirs(log_dir, exist_ok=True)
    
    # Wrap the environment with SB3 Monitor to log episode reward, length, and wall-clock time to CSV
    env = Monitor(env, filename=os.path.join(log_dir, 'monitor.csv'))

    # Return the fully configured and wrapped environment
    return env