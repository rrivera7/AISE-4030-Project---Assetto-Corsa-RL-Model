class SACAgent:
    def __init__(self, env, config):
        """
        Initializes the SAC agent using Stable-Baselines3.
        
        Library Parameters Documented: More parameters given here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html#parameters but the most important ones for this project in my opinion are:
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
            evaluate: If True, uses deterministic actions (exploitation). 
                             If False, adds stochastic noise (exploration).
                             
        Returns:
            numpy.ndarray: The chosen continuous action array.
        """
        pass

    def train(self, total_timesteps, callback=None):
        """
        Executes the learning loop using SB3's built-in .learn() method. May only be a one line call to the SB3 API, but this method serves as the main entry point for training the agent.
        
        Args:
            total_timesteps (int): The total number of environment steps to train for.
            callback: SB3 callback for logging or saving.
            
        Returns:
            None
        """
        pass

    def save_model(self, filepath):
        """
        Saves the SB3 model weights and configuration to disk. May only be a one line call to the SB3 API, but this method abstracts away the saving logic and allows for future extensions (e.g., saving additional metadata).
        
        Args:
            filepath (str): The desired full path for the saved .zip file.
            
        Returns:
            None
        """
        pass

    def load_model(self, filepath):
        """
        Loads a trained SB3 SAC model from disk. Since there is potentially a save_model method, there should be a corresponding load_model method.
        
        Args:
            filepath (str): The full path to the saved model .zip file.
            
        Returns:
            None
        """
        pass