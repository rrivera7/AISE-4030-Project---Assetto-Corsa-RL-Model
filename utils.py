import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
# Use the 'Agg' backend for matplotlib so it doesn't try to open windows (useful for headless servers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class ModelCallback(BaseCallback):
    """
    Combined checkpoint-saving and metric-logging callback for SB3 SAC.

    At every ``check_freq`` steps:
      - Saves a full model checkpoint (weights + replay buffer + metadata).
    At every ``log_loss_freq`` gradient steps:
      - Appends actor loss, critic loss, entropy coefficient, and
        entropy coefficient loss to a CSV.
    At the end of every episode:
      - Appends episode reward and episode length to a separate CSV.
    At every ``episode_checkpoint_freq`` episodes:
      - Saves a versioned checkpoint (checkpoint_ep50, checkpoint_ep100, etc.).

    Metric CSVs are written to ``results_dir`` (SAC_Results/) so they
    can be loaded for plotting without retraining.  Checkpoints are
    written to ``save_path``.

    Args:
        check_freq (int): Save a model checkpoint every this many env steps.
        log_loss_freq (int): Write loss/entropy row every this many env steps.
        save_path (str): Directory for checkpoints.
        results_dir (str): Directory for metric CSV files.
        episode_checkpoint_freq (int): Save a versioned checkpoint every this many episodes.
        verbose (int): Verbosity level.
    """

    def __init__(self, check_freq: int, log_loss_freq: int, save_path: str,
                 results_dir: str = "./SAC_Results",
                 episode_checkpoint_freq: int = 50, verbose=0):
        # Initialize the base callback with the specified verbosity level
        super(ModelCallback, self).__init__(verbose)
        
        # Store configuration parameters
        self.check_freq = check_freq
        self.log_loss_freq = log_loss_freq
        self.save_path = save_path
        self.results_dir = results_dir
        self.episode_checkpoint_freq = episode_checkpoint_freq

        # Paths for the two CSV log files (in results_dir)
        self.loss_csv_path = os.path.join(results_dir, "loss_log.csv")
        self.episode_csv_path = os.path.join(results_dir, "episode_log.csv")

        # Running episode counter (avoids re-reading CSV)
        self._episode_count = 0

    def _init_callback(self) -> None:
        """
        Called once before training starts.  Creates output directories
        and writes CSV headers if the files don't already exist.
        If resuming, counts existing episodes so the counter stays accurate.

        Returns:
            None
        """
        # Ensure the directories for checkpoints and results exist
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Write loss CSV header if the file is new
        if not os.path.exists(self.loss_csv_path):
            with open(self.loss_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "actor_loss", "critic_loss",
                    "ent_coef", "ent_coef_loss"
                ])

        # Write episode CSV header (or resume counter from existing file)
        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "episode", "episode_reward", "episode_length"
                ])
        else:
            # Count existing rows so episode numbering continues correctly
            with open(self.episode_csv_path, 'r') as f:
                self._episode_count = max(0, sum(1 for _ in f) - 1)

    def _on_step(self) -> bool:
        """
        Called at every environment step.

        Returns:
            bool: True to continue training.
        """
        # --- Full checkpoint saving (model + replay buffer + metadata) ---
        if self.n_calls % self.check_freq == 0:
            # Define the path for the latest checkpoint
            checkpoint_dir = os.path.join(self.save_path, "checkpoint_latest")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save the model weights
            model_path = os.path.join(checkpoint_dir, "model")
            self.model.save(model_path)

            # Save the replay buffer
            buffer_path = os.path.join(checkpoint_dir, "replay_buffer")
            self.model.save_replay_buffer(buffer_path)

            # Save training metadata to a JSON file
            meta = {
                "num_timesteps": self.model.num_timesteps,
                "num_episodes": self.model._episode_num,
                "n_calls": self.n_calls,
            }
            meta_path = os.path.join(checkpoint_dir, "training_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            # Print a message if verbosity is enabled
            if self.verbose > 0:
                print(f"Full checkpoint saved: {checkpoint_dir}")

        # --- Loss / entropy logging ---
        if self.n_calls % self.log_loss_freq == 0:
            # Retrieve the current values from the model's logger
            logger = self.model.logger.name_to_value
            actor_loss = logger.get("train/actor_loss", float('nan'))
            critic_loss = logger.get("train/critic_loss", float('nan'))
            ent_coef = logger.get("train/ent_coef", float('nan'))
            ent_coef_loss = logger.get("train/ent_coef_loss", float('nan'))

            # Append the retrieved values to the loss CSV
            with open(self.loss_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, actor_loss, critic_loss,
                    ent_coef, ent_coef_loss
                ])

        # --- Episode-level logging ---
        # SB3 Monitor wrapper stores completed episode info in self.locals
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                # Increment the episode counter
                self._episode_count += 1
                
                # Append the episode statistics to the episode CSV
                with open(self.episode_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        self._episode_count,
                        ep_info["r"],
                        ep_info["l"],
                    ])

                # --- Episode-based versioned checkpoint ---
                if self._episode_count % self.episode_checkpoint_freq == 0:
                    # Define the path for the versioned checkpoint
                    ep_ckpt_dir = os.path.join(
                        self.save_path,
                        f"checkpoint_ep{self._episode_count}"
                    )
                    os.makedirs(ep_ckpt_dir, exist_ok=True)
                    
                    # Save the model and replay buffer
                    self.model.save(os.path.join(ep_ckpt_dir, "model"))
                    self.model.save_replay_buffer(os.path.join(ep_ckpt_dir, "replay_buffer"))
                    
                    # Save the metadata
                    meta = {
                        "num_timesteps": self.model.num_timesteps,
                        "num_episodes": self._episode_count,
                        "n_calls": self.n_calls,
                    }
                    with open(os.path.join(ep_ckpt_dir, "training_meta.json"), 'w') as f:
                        json.dump(meta, f)
                    print(f"Episode checkpoint saved: {ep_ckpt_dir}")

        # Return True to indicate that training should continue
        return True


class PPOModelCallback(BaseCallback):
    """
    Combined checkpoint-saving and metric-logging callback for SB3 PPO.

    Same structure as ModelCallback but adapted for on-policy PPO:
      - No replay buffer saving (PPO is on-policy).
      - Logs PPO-specific loss keys: policy_gradient_loss, value_loss,
        entropy_loss, approx_kl, clip_fraction.

    Args:
        check_freq (int): Save a model checkpoint every this many env steps.
        log_loss_freq (int): Write loss row every this many env steps.
        save_path (str): Directory for checkpoints.
        results_dir (str): Directory for metric CSV files.
        episode_checkpoint_freq (int): Save a versioned checkpoint every this many episodes.
        verbose (int): Verbosity level.
    """

    def __init__(self, check_freq: int, log_loss_freq: int, save_path: str,
                 results_dir: str = "./PPO_Results",
                 episode_checkpoint_freq: int = 50, verbose=0):
        # Initialize the base callback with the specified verbosity level
        super(PPOModelCallback, self).__init__(verbose)
        
        # Store configuration parameters
        self.check_freq = check_freq
        self.log_loss_freq = log_loss_freq
        self.save_path = save_path
        self.results_dir = results_dir
        self.episode_checkpoint_freq = episode_checkpoint_freq

        # Paths for the two CSV log files
        self.loss_csv_path = os.path.join(results_dir, "loss_log.csv")
        self.episode_csv_path = os.path.join(results_dir, "episode_log.csv")
        
        # Running episode counter
        self._episode_count = 0

    def _init_callback(self) -> None:
        """
        Called once before training starts. Creates output directories and writes CSV headers.
        """
        # Ensure the directories for checkpoints and results exist
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Write loss CSV header if the file is new
        if not os.path.exists(self.loss_csv_path):
            with open(self.loss_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "policy_gradient_loss", "value_loss",
                    "entropy_loss", "approx_kl", "clip_fraction"
                ])

        # Write episode CSV header (or resume counter from existing file)
        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "episode", "episode_reward", "episode_length"
                ])
        else:
            # Count existing rows so episode numbering continues correctly
            with open(self.episode_csv_path, 'r') as f:
                self._episode_count = max(0, sum(1 for _ in f) - 1)

    def _on_step(self) -> bool:
        """
        Called at every environment step.

        Returns:
            bool: True to continue training.
        """
        # --- Checkpoint saving (model + metadata) ---
        if self.n_calls % self.check_freq == 0:
            # Define the path for the latest checkpoint
            checkpoint_dir = os.path.join(self.save_path, "checkpoint_latest")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save the model weights
            self.model.save(os.path.join(checkpoint_dir, "model"))

            # Save training metadata to a JSON file
            meta = {
                "num_timesteps": self.model.num_timesteps,
                "num_episodes": getattr(self.model, '_episode_num', 0),
                "n_calls": self.n_calls,
            }
            with open(os.path.join(checkpoint_dir, "training_meta.json"), 'w') as f:
                json.dump(meta, f)

            # Print a message if verbosity is enabled
            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_dir}")

        # --- Loss logging ---
        if self.n_calls % self.log_loss_freq == 0:
            # Retrieve the current values from the model's logger
            logger = self.model.logger.name_to_value
            pg_loss = logger.get("train/policy_gradient_loss", float('nan'))
            value_loss = logger.get("train/value_loss", float('nan'))
            entropy_loss = logger.get("train/entropy_loss", float('nan'))
            approx_kl = logger.get("train/approx_kl", float('nan'))
            clip_fraction = logger.get("train/clip_fraction", float('nan'))

            # Append the retrieved values to the loss CSV
            with open(self.loss_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, pg_loss, value_loss,
                    entropy_loss, approx_kl, clip_fraction
                ])

        # --- Episode-level logging ---
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                # Increment the episode counter
                self._episode_count += 1
                
                # Append the episode statistics to the episode CSV
                with open(self.episode_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        self._episode_count,
                        ep_info["r"],
                        ep_info["l"],
                    ])

                # --- Episode-based versioned checkpoint ---
                if self._episode_count % self.episode_checkpoint_freq == 0:
                    # Define the path for the versioned checkpoint
                    ep_ckpt_dir = os.path.join(
                        self.save_path,
                        f"checkpoint_ep{self._episode_count}"
                    )
                    os.makedirs(ep_ckpt_dir, exist_ok=True)
                    
                    # Save the model weights
                    self.model.save(os.path.join(ep_ckpt_dir, "model"))
                    
                    # Save the metadata
                    meta = {
                        "num_timesteps": self.model.num_timesteps,
                        "num_episodes": self._episode_count,
                        "n_calls": self.n_calls,
                    }
                    with open(os.path.join(ep_ckpt_dir, "training_meta.json"), 'w') as f:
                        json.dump(meta, f)
                    print(f"Episode checkpoint saved: {ep_ckpt_dir}")

        # Return True to indicate that training should continue
        return True


def plot_learning_curve(results_dir, title="Learning Curve", window_size=50):
    """
    Generates a smoothed reward-over-episodes plot from the episode_log.csv
    in the results directory. Saves the figure as a PNG file.

    Args:
        results_dir (str): Directory containing episode_log.csv.
        title (str): Plot title.
        window_size (int): Rolling-average window for smoothing.

    Returns:
        None
    """
    # Define the path to the episode log CSV
    csv_path = os.path.join(results_dir, "episode_log.csv")
    try:
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Handle the case where the file doesn't exist
        print(f"No episode log found at {csv_path}")
        return

    # Check if the DataFrame is empty
    if df.empty:
        print("Episode log has no entries yet.")
        return

    # Calculate the rolling mean and standard deviation for smoothing
    smoothed = df['episode_reward'].rolling(window=window_size, min_periods=1).mean()
    raw_std = df['episode_reward'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Initialize the plot figure
    plt.figure(figsize=(10, 5))
    
    # Plot the raw reward data with low opacity
    plt.plot(df['episode'], df['episode_reward'], alpha=0.2, color='steelblue', label='Raw Reward')
    
    # Plot the smoothed reward data
    plt.plot(df['episode'], smoothed, color='steelblue', linewidth=2,
             label=f'Smoothed (window={window_size})')
             
    # Add a shaded region representing the standard deviation
    plt.fill_between(df['episode'], smoothed - raw_std, smoothed + raw_std,
                     alpha=0.15, color='steelblue')
                     
    # Set plot titles and labels
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot to a PNG file
    plot_path = os.path.join(results_dir, "learning_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to {plot_path}")


def plot_loss_curves(results_dir, window_size=50):
    """
    Generates actor-loss, critic-loss, entropy-coefficient, and
    entropy-coefficient-loss plots from the loss_log.csv in results dir.

    Args:
        results_dir (str): Directory containing loss_log.csv.
        window_size (int): Rolling-average window for smoothing.

    Returns:
        None
    """
    # Define the path to the loss log CSV
    csv_path = os.path.join(results_dir, "loss_log.csv")
    try:
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Handle the case where the file doesn't exist
        print(f"No loss log found at {csv_path}")
        return

    # Drop rows where losses haven't been recorded yet (before learning_starts)
    df = df.dropna(subset=["actor_loss", "critic_loss"])
    
    # Check if the DataFrame is empty after dropping NaNs
    if df.empty:
        print("Loss log has no valid entries yet.")
        return

    # Create a figure with 4 subplots sharing the x-axis
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # Iterate over each metric and its corresponding subplot
    for ax, col, label, color in [
        (axes[0], "actor_loss", "Actor (Policy) Loss", "tab:blue"),
        (axes[1], "critic_loss", "Critic (Q-Network) Loss", "tab:red"),
        (axes[2], "ent_coef", "Entropy Coefficient (alpha)", "tab:green"),
        (axes[3], "ent_coef_loss", "Entropy Coef. Loss", "tab:orange"),
    ]:
        # Extract the raw data for the metric
        raw = df[col].astype(float)
        
        # Calculate the rolling mean for smoothing
        smoothed = raw.rolling(window=window_size, min_periods=1).mean()
        
        # Plot the raw and smoothed data
        ax.plot(df["timestep"], raw, alpha=0.2, color=color)
        ax.plot(df["timestep"], smoothed, color=color, linewidth=2)
        
        # Set the y-axis label and enable the grid
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Set the x-axis label for the bottom subplot
    axes[-1].set_xlabel("Timestep")
    
    # Set the overall figure title
    fig.suptitle("SAC Training Losses", fontsize=14)
    plt.tight_layout()

    # Save the plot to a PNG file
    plot_path = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {plot_path}")


def plot_ppo_loss_curves(results_dir, window_size=50):
    """
    Generates PPO-specific loss plots from loss_log.csv:
    policy gradient loss, value loss, entropy loss, approx KL, clip fraction.

    Args:
        results_dir (str): Directory containing loss_log.csv.
        window_size (int): Rolling-average window for smoothing.

    Returns:
        None
    """
    # Define the path to the loss log CSV
    csv_path = os.path.join(results_dir, "loss_log.csv")
    try:
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Handle the case where the file doesn't exist
        print(f"No loss log found at {csv_path}")
        return

    # Drop rows with missing values for key metrics
    df = df.dropna(subset=["policy_gradient_loss", "value_loss"])
    
    # Check if the DataFrame is empty after dropping NaNs
    if df.empty:
        print("Loss log has no valid entries yet.")
        return

    # Define the metrics to plot
    metrics = [
        ("policy_gradient_loss", "Policy Gradient Loss", "tab:blue"),
        ("value_loss", "Value Loss", "tab:red"),
        ("entropy_loss", "Entropy Loss", "tab:green"),
        ("approx_kl", "Approx KL Divergence", "tab:orange"),
        ("clip_fraction", "Clip Fraction", "tab:purple"),
    ]

    # Create a figure with a subplot for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.5 * len(metrics)), sharex=True)

    # Iterate over each metric and its corresponding subplot
    for ax, (col, label, color) in zip(axes, metrics):
        # Extract the raw data for the metric
        raw = df[col].astype(float)
        
        # Calculate the rolling mean for smoothing
        smoothed = raw.rolling(window=window_size, min_periods=1).mean()
        
        # Plot the raw and smoothed data
        ax.plot(df["timestep"], raw, alpha=0.2, color=color)
        ax.plot(df["timestep"], smoothed, color=color, linewidth=2)
        
        # Set the y-axis label and enable the grid
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Set the x-axis label for the bottom subplot
    axes[-1].set_xlabel("Timestep")
    
    # Set the overall figure title
    fig.suptitle("PPO Training Losses", fontsize=14)
    plt.tight_layout()

    # Save the plot to a PNG file
    plot_path = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"PPO loss curves saved to {plot_path}")


# =====================================================================
#  Task 3 – Comparative Analysis Plots (SAC vs PPO)
# =====================================================================

def _count_episode_rows(csv_path):
    """
    Counts the number of data rows before the first blank line in an
    eval_results.csv file. The blank line separates per-episode rows
    from the aggregate summary row at the bottom.

    Args:
        csv_path (str): Path to the eval_results.csv file.

    Returns:
        int: Number of per-episode data rows (excluding the header).
    """
    count = 0
    with open(csv_path, 'r') as f:
        next(f)  # skip the header row
        for line in f:
            if line.strip() == "":
                break
            count += 1
    return count


def _load_episodes(results_dir):
    """
    Helper function to load episode log data from a specific directory.
    
    Args:
        results_dir (str): Directory containing the episode_log.csv file.
        
    Returns:
        pd.DataFrame: The loaded episode data with appropriate types.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(results_dir, "episode_log.csv"))
    
    # Ensure specific columns are treated as floats
    df["episode_reward"] = df["episode_reward"].astype(float)
    df["timestep"] = df["timestep"].astype(float)
    
    return df


def plot_comparative_learning_speed(
    sac_dir="./SAC_Results",
    ppo_dir="./PPO_Results",
    out_dir="./Comparative_Plots",
    window_size=50,
    reward_threshold=0.0,
):
    """
    Metric 1 – Learning Speed.
    Overlays both agents' smoothed reward curves vs timestep on the same plot.
    Marks with a vertical line where each agent first crosses the reward threshold.
    
    Args:
        sac_dir (str): Directory containing SAC results.
        ppo_dir (str): Directory containing PPO results.
        out_dir (str): Directory to save the generated plot.
        window_size (int): Rolling-average window for smoothing.
        reward_threshold (float): Target reward value to mark on the plot.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the episode data for both algorithms
    sac = _load_episodes(sac_dir)
    ppo = _load_episodes(ppo_dir)

    # Calculate smoothed reward curves
    sac_smooth = sac["episode_reward"].rolling(window=window_size, min_periods=1).mean()
    ppo_smooth = ppo["episode_reward"].rolling(window=window_size, min_periods=1).mean()

    # Initialize the plot figure
    plt.figure(figsize=(12, 6))
    
    # Plot the smoothed curves
    plt.plot(sac["timestep"], sac_smooth, color="tab:blue", linewidth=2, label="SAC")
    plt.plot(ppo["timestep"], ppo_smooth, color="tab:orange", linewidth=2, label="PPO")

    # Add vertical lines where each curve crosses the reward threshold
    for smooth, ts, name, color, ls in [
        (sac_smooth, sac["timestep"], "SAC", "tab:blue", "--"),
        (ppo_smooth, ppo["timestep"], "PPO", "tab:orange", "--"),
    ]:
        # Find all points where the smoothed reward is above the threshold
        crossings = ts[smooth >= reward_threshold]
        if not crossings.empty:
            # Get the first crossing point
            first = crossings.iloc[0]
            
            # Draw a vertical line at the crossing point
            plt.axvline(first, color=color, linestyle=ls, alpha=0.7)
            
            # Add a text label indicating the crossing point
            plt.text(first, plt.ylim()[1] * 0.95, f" {name} ≥ {reward_threshold}\n step {int(first)}",
                     color=color, fontsize=8, va="top")

    # Draw a horizontal line for the reward threshold
    plt.axhline(reward_threshold, color="gray", linestyle=":", alpha=0.5, label=f"Threshold = {reward_threshold}")
    
    # Set plot labels and title
    plt.xlabel("Timestep")
    plt.ylabel(f"Episode Reward (rolling mean, w={window_size})")
    plt.title("Learning Speed: SAC vs PPO")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    path = os.path.join(out_dir, "learning_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Learning speed plot saved to {path}")


def plot_comparative_loss_convergence(
    sac_dir="./SAC_Results",
    ppo_dir="./PPO_Results",
    out_dir="./Comparative_Plots",
    window_size=100,
):
    """
    Metric 2 – Loss Convergence.
    Top panel:  SAC critic loss vs PPO value loss (both approximate the value function).
    Bottom panel: SAC actor loss vs PPO policy gradient loss.
    Smoothed with rolling mean to reveal trends.
    
    Args:
        sac_dir (str): Directory containing SAC results.
        ppo_dir (str): Directory containing PPO results.
        out_dir (str): Directory to save the generated plot.
        window_size (int): Rolling-average window for smoothing.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Load the loss data for both algorithms
    sac_loss = pd.read_csv(os.path.join(sac_dir, "loss_log.csv"))
    ppo_loss = pd.read_csv(os.path.join(ppo_dir, "loss_log.csv"))

    # Drop rows with missing values
    sac_loss = sac_loss.dropna(subset=["critic_loss"])
    ppo_loss = ppo_loss.dropna(subset=["value_loss"])

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

    # --- Top panel: Value / critic loss ---
    ax = axes[0]
    
    # Calculate smoothed curves
    sac_cl = sac_loss["critic_loss"].astype(float).rolling(window=window_size, min_periods=1).mean()
    ppo_vl = ppo_loss["value_loss"].astype(float).rolling(window=window_size, min_periods=1).mean()
    
    # Plot the smoothed curves
    ax.plot(sac_loss["timestep"], sac_cl, color="tab:blue", linewidth=2, label="SAC Critic Loss")
    ax.plot(ppo_loss["timestep"], ppo_vl, color="tab:orange", linewidth=2, label="PPO Value Loss")
    
    # Set labels and title
    ax.set_ylabel("Loss")
    ax.set_title("Value-Function Loss Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom panel: Policy loss ---
    ax = axes[1]
    
    # Calculate smoothed curves
    sac_al = sac_loss["actor_loss"].astype(float).rolling(window=window_size, min_periods=1).mean()
    ppo_pg = ppo_loss["policy_gradient_loss"].astype(float).rolling(window=window_size, min_periods=1).mean()
    
    # Plot the smoothed curves
    ax.plot(sac_loss["timestep"], sac_al, color="tab:blue", linewidth=2, label="SAC Actor Loss")
    ax.plot(ppo_loss["timestep"], ppo_pg, color="tab:orange", linewidth=2, label="PPO Policy Gradient Loss")
    
    # Set labels and title
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set overall figure title
    fig.suptitle("Loss Convergence: SAC vs PPO", fontsize=14, y=1.01)
    plt.tight_layout()
    
    # Save the plot
    path = os.path.join(out_dir, "loss_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss convergence plot saved to {path}")


def plot_comparative_final_performance(
    sac_dir="./SAC_Results",
    ppo_dir="./PPO_Results",
    out_dir="./Comparative_Plots",
):
    """
    Metric 3 – Final Performance.
    Bar chart of mean reward ± std using the eval_results.csv from each
    algorithm's results directory (generated during deployment/evaluation).
    Prints a summary table to the console as well.

    Args:
        sac_dir (str): Directory containing SAC eval_results.csv.
        ppo_dir (str): Directory containing PPO eval_results.csv.
        out_dir (str): Directory to save the generated plot.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Load the per-episode evaluation results for both algorithms
    sac_csv = os.path.join(sac_dir, "eval_results.csv")
    ppo_csv = os.path.join(ppo_dir, "eval_results.csv")

    # Read only the per-episode rows (stop before the blank separator line)
    sac_eval = pd.read_csv(sac_csv, nrows=_count_episode_rows(sac_csv))
    ppo_eval = pd.read_csv(ppo_csv, nrows=_count_episode_rows(ppo_csv))

    # Extract the reward columns as floats
    sac_rewards = sac_eval["reward"].astype(float)
    ppo_rewards = ppo_eval["reward"].astype(float)

    # Calculate means and standard deviations
    means = [sac_rewards.mean(), ppo_rewards.mean()]
    stds = [sac_rewards.std(), ppo_rewards.std()]

    # Number of evaluation episodes used
    n_sac = len(sac_rewards)
    n_ppo = len(ppo_rewards)

    # Define labels and colors for the bar chart
    labels = ["SAC", "PPO"]
    colors = ["tab:blue", "tab:orange"]

    # Print summary table to console
    print(f"\n{'='*50}")
    print(f"  Final Deployment Performance (Eval Results)")
    print(f"{'='*50}")
    print(f"  SAC  ({n_sac:3d} eps):  {means[0]:+.2f}  ±  {stds[0]:.2f}")
    print(f"  PPO  ({n_ppo:3d} eps):  {means[1]:+.2f}  ±  {stds[1]:.2f}")
    print(f"{'='*50}\n")

    # Initialize the plot figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw the bar chart with error bars
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.85, edgecolor="black")

    # Add text labels above each bar
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 2,
                f"{m:.1f} ± {s:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Set labels and title
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"Final Deployment Performance\n(SAC: {n_sac} eps, PPO: {n_ppo} eps — from eval_results.csv)")

    # Draw a horizontal line at y=0
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # Save the plot
    path = os.path.join(out_dir, "final_performance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Final performance plot saved to {path}")


def plot_comparative_stability(
    sac_dir="./SAC_Results",
    ppo_dir="./PPO_Results",
    out_dir="./Comparative_Plots",
    window_size=50,
):
    """
    Metric 4 – Stability / Variance.
    Overlays smoothed reward curves with shaded ±1 std rolling bands for both agents.
    
    Args:
        sac_dir (str): Directory containing SAC results.
        ppo_dir (str): Directory containing PPO results.
        out_dir (str): Directory to save the generated plot.
        window_size (int): Rolling-average window for smoothing.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the episode data for both algorithms
    sac = _load_episodes(sac_dir)
    ppo = _load_episodes(ppo_dir)

    # Initialize the plot figure
    plt.figure(figsize=(12, 6))

    # Plot data for both algorithms
    for df, name, color in [(sac, "SAC", "tab:blue"), (ppo, "PPO", "tab:orange")]:
        # Calculate smoothed mean and standard deviation
        smooth = df["episode_reward"].rolling(window=window_size, min_periods=1).mean()
        rstd = df["episode_reward"].rolling(window=window_size, min_periods=1).std().fillna(0)
        ts = df["timestep"]
        
        # Plot the smoothed mean curve
        plt.plot(ts, smooth, color=color, linewidth=2, label=f"{name} Mean")
        
        # Add shaded region for ±1 standard deviation
        plt.fill_between(ts, smooth - rstd, smooth + rstd, alpha=0.15, color=color, label=f"{name} ± 1 std")

    # Set labels and title
    plt.xlabel("Timestep")
    plt.ylabel(f"Episode Reward (rolling w={window_size})")
    plt.title("Stability / Variance: SAC vs PPO")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    path = os.path.join(out_dir, "stability_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Stability / variance plot saved to {path}")


def generate_all_comparative_plots(
    sac_dir="./SAC_Results",
    ppo_dir="./PPO_Results",
    out_dir="./Comparative_Plots",
    window_size=50,
    reward_threshold=0.0,
):
    """
    One-call convenience function that generates all four Task 3 comparative plots.
    
    Args:
        sac_dir (str): Directory containing SAC results.
        ppo_dir (str): Directory containing PPO results.
        out_dir (str): Directory to save the generated plots.
        window_size (int): Rolling-average window for smoothing.
        reward_threshold (float): Target reward value to mark on the learning speed plot.
    """
    print("\n=== Generating Task 3 Comparative Plots ===\n")
    
    # Generate Learning Speed plot
    plot_comparative_learning_speed(sac_dir, ppo_dir, out_dir, window_size, reward_threshold)
    
    # Generate Loss Convergence plot
    plot_comparative_loss_convergence(sac_dir, ppo_dir, out_dir, window_size=100)
    
    # Generate Final Performance plot (uses eval_results.csv from each directory)
    plot_comparative_final_performance(sac_dir, ppo_dir, out_dir)
    
    # Generate Stability / Variance plot
    plot_comparative_stability(sac_dir, ppo_dir, out_dir, window_size)
    
    print(f"\nAll comparative plots saved to {out_dir}/")