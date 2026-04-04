import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomTelemetryExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Assetto Corsa telemetry data.
    Takes the raw flat observation vector and projects it through a small
    two-layer MLP to produce a learned feature representation before
    it reaches the SAC actor and critic heads.

    Args:
        observation_space (gym.spaces.Box): The environment observation space.
        features_dim (int): Dimensionality of the output feature vector.
                            Controlled by config.policy.features_dim.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomTelemetryExtractor, self).__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations (torch.Tensor): Batch of flat observation vectors.

        Returns:
            torch.Tensor: Learned feature representation of shape (batch, features_dim).
        """
        return self.extractor(observations)