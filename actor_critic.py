import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomTelemetryExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor network that processes Assetto Corsa data
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
        Performs a forward pass through the feature extractor. May not be required for the implementation of the SAC agent bec of SB3 library but keeping here 
        just in case... it is somethuing that falls under {type}_netwrok.py file in the phase 2 doc (part of the NN architecture)
        
        Args:
            observations: The state vector from the environment.
            
        Returns:
            feature_shape_array: The extracted features of shape (batch_size, features_dim).
        """
        pass