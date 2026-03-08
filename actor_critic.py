"""
actor_critic.py
Responsibility: Custom neural network feature extractors for use with SB3's SAC.
"""
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomTelemetryExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor network that processes Assetto Corsa telemetry 
    before feeding it into SB3's internal Actor and Critic networks.
    """
    def __init__(self, observation_space, features_dim=256):
        """
        Initializes the custom feature extraction layers.
        
        Args:
            observation_space (gym.Space): The environment's observation space.
            features_dim (int): The number of output features to pass to the policy.
            
        Returns:
            None
        """
        pass

    def forward(self, observations):
        """
        Performs a forward pass through the feature extractor.
        
        Args:
            observations (torch.Tensor): The 125-dimensional state vector from the environment.
            
        Returns:
            torch.Tensor: The extracted features of shape (batch_size, features_dim).
        """
        pass