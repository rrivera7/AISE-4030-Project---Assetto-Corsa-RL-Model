"""
sac_agent.py
Responsibility: Wrapper for the Stable-Baselines3 Soft Actor-Critic (SAC) algorithm.
"""

class SACAgent:
    """
    A wrapper class for the Stable-Baselines3 SAC implementation.
    """
    def __init__(self, env, config):
        """
        Initializes the SAC agent using Stable-Baselines3.
        
        Library Parameters Documented:
        - policy (str): Defines the network architecture (e.g., 'MlpPolicy').
        - learning_rate (float): The step size for network updates.
        - buffer_size (int): The maximum capacity of the internal replay buffer.
        - batch_size (int): Number of experiences sampled per update.
        - gamma (float): The discount factor for future rewards.
        - tau (float): The soft update coefficient for target networks.
        
        Args:
            env (gym.Env): The Assetto Corsa Gym environment instance.
            config (dict): A dictionary of hyperparameters loaded from config.yaml.
            
        Returns:
            None
        """
        pass

    def select_action(self, state, evaluate=False):
        """
        Chooses an action for a given state using the SB3 model's predict method.
        
        Args:
            state (numpy.ndarray): The current state representation from the environment.
            evaluate (bool): If True, uses deterministic actions (exploitation). 
                             If False, adds stochastic noise (exploration).
                             
        Returns:
            numpy.ndarray: The chosen continuous action array.
        """
        pass

    def train(self, total_timesteps, callback=None):
        """
        Executes the learning loop using SB3's built-in .learn() method.
        
        Args:
            total_timesteps (int): The total number of environment steps to train for.
            callback (BaseCallback, optional): SB3 callback for logging or saving.
            
        Returns:
            None
        """
        pass

    def save_model(self, filepath):
        """
        Saves the SB3 model weights and configuration to disk.
        
        Args:
            filepath (str): The desired full path for the saved .zip file.
            
        Returns:
            None
        """
        pass

    def load_model(self, filepath):
        """
        Loads a trained SB3 SAC model from disk.
        
        Args:
            filepath (str): The full path to the saved model .zip file.
            
        Returns:
            None
        """
        pass