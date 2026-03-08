import yaml
import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# --- Utility Functions ---
# ===================================================================
def load_config(filepath):
    """
    Loads the YAML configuration file.

    Args:
        filepath (str): Path to the config.yaml file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def plot_results(rewards, losses, save_path, moving_avg_window):
    """
    Plots the learning curves (returns and losses) as required by Phase 1 metrics[cite: 100, 104].

    Args:
        rewards (list): List of episodic returns.
        losses (list): List of episodic average losses.
        save_path (str): Filepath to save the plot.
        moving_avg_window (int): Window for the rolling mean.
    """
    pass # Skeleton: Implement matplotlib plotting logic