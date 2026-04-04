import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
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
        super(ModelCallback, self).__init__(verbose)
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
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Write loss CSV header
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
            checkpoint_dir = os.path.join(self.save_path, "checkpoint_latest")
            os.makedirs(checkpoint_dir, exist_ok=True)

            model_path = os.path.join(checkpoint_dir, "model")
            self.model.save(model_path)

            buffer_path = os.path.join(checkpoint_dir, "replay_buffer")
            self.model.save_replay_buffer(buffer_path)

            meta = {
                "num_timesteps": self.model.num_timesteps,
                "num_episodes": self.model._episode_num,
                "n_calls": self.n_calls,
            }
            meta_path = os.path.join(checkpoint_dir, "training_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            if self.verbose > 0:
                print(f"Full checkpoint saved: {checkpoint_dir}")

        # --- Loss / entropy logging ---
        if self.n_calls % self.log_loss_freq == 0:
            logger = self.model.logger.name_to_value
            actor_loss = logger.get("train/actor_loss", float('nan'))
            critic_loss = logger.get("train/critic_loss", float('nan'))
            ent_coef = logger.get("train/ent_coef", float('nan'))
            ent_coef_loss = logger.get("train/ent_coef_loss", float('nan'))

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
                self._episode_count += 1
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
                    ep_ckpt_dir = os.path.join(
                        self.save_path,
                        f"checkpoint_ep{self._episode_count}"
                    )
                    os.makedirs(ep_ckpt_dir, exist_ok=True)
                    self.model.save(os.path.join(ep_ckpt_dir, "model"))
                    self.model.save_replay_buffer(os.path.join(ep_ckpt_dir, "replay_buffer"))
                    meta = {
                        "num_timesteps": self.model.num_timesteps,
                        "num_episodes": self._episode_count,
                        "n_calls": self.n_calls,
                    }
                    with open(os.path.join(ep_ckpt_dir, "training_meta.json"), 'w') as f:
                        json.dump(meta, f)
                    print(f"Episode checkpoint saved: {ep_ckpt_dir}")

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
        super(PPOModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_loss_freq = log_loss_freq
        self.save_path = save_path
        self.results_dir = results_dir
        self.episode_checkpoint_freq = episode_checkpoint_freq

        self.loss_csv_path = os.path.join(results_dir, "loss_log.csv")
        self.episode_csv_path = os.path.join(results_dir, "episode_log.csv")
        self._episode_count = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        if not os.path.exists(self.loss_csv_path):
            with open(self.loss_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "policy_gradient_loss", "value_loss",
                    "entropy_loss", "approx_kl", "clip_fraction"
                ])

        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "episode", "episode_reward", "episode_length"
                ])
        else:
            with open(self.episode_csv_path, 'r') as f:
                self._episode_count = max(0, sum(1 for _ in f) - 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            checkpoint_dir = os.path.join(self.save_path, "checkpoint_latest")
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.model.save(os.path.join(checkpoint_dir, "model"))

            meta = {
                "num_timesteps": self.model.num_timesteps,
                "num_episodes": getattr(self.model, '_episode_num', 0),
                "n_calls": self.n_calls,
            }
            with open(os.path.join(checkpoint_dir, "training_meta.json"), 'w') as f:
                json.dump(meta, f)

            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_dir}")

        if self.n_calls % self.log_loss_freq == 0:
            logger = self.model.logger.name_to_value
            pg_loss = logger.get("train/policy_gradient_loss", float('nan'))
            value_loss = logger.get("train/value_loss", float('nan'))
            entropy_loss = logger.get("train/entropy_loss", float('nan'))
            approx_kl = logger.get("train/approx_kl", float('nan'))
            clip_fraction = logger.get("train/clip_fraction", float('nan'))

            with open(self.loss_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, pg_loss, value_loss,
                    entropy_loss, approx_kl, clip_fraction
                ])

        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self._episode_count += 1
                with open(self.episode_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        self._episode_count,
                        ep_info["r"],
                        ep_info["l"],
                    ])

                if self._episode_count % self.episode_checkpoint_freq == 0:
                    ep_ckpt_dir = os.path.join(
                        self.save_path,
                        f"checkpoint_ep{self._episode_count}"
                    )
                    os.makedirs(ep_ckpt_dir, exist_ok=True)
                    self.model.save(os.path.join(ep_ckpt_dir, "model"))
                    meta = {
                        "num_timesteps": self.model.num_timesteps,
                        "num_episodes": self._episode_count,
                        "n_calls": self.n_calls,
                    }
                    with open(os.path.join(ep_ckpt_dir, "training_meta.json"), 'w') as f:
                        json.dump(meta, f)
                    print(f"Episode checkpoint saved: {ep_ckpt_dir}")

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
    csv_path = os.path.join(results_dir, "episode_log.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"No episode log found at {csv_path}")
        return

    if df.empty:
        print("Episode log has no entries yet.")
        return

    smoothed = df['episode_reward'].rolling(window=window_size, min_periods=1).mean()
    raw_std = df['episode_reward'].rolling(window=window_size, min_periods=1).std().fillna(0)

    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['episode_reward'], alpha=0.2, color='steelblue', label='Raw Reward')
    plt.plot(df['episode'], smoothed, color='steelblue', linewidth=2,
             label=f'Smoothed (window={window_size})')
    plt.fill_between(df['episode'], smoothed - raw_std, smoothed + raw_std,
                     alpha=0.15, color='steelblue')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

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
    csv_path = os.path.join(results_dir, "loss_log.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"No loss log found at {csv_path}")
        return

    # Drop rows where losses haven't been recorded yet (before learning_starts)
    df = df.dropna(subset=["actor_loss", "critic_loss"])
    if df.empty:
        print("Loss log has no valid entries yet.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    for ax, col, label, color in [
        (axes[0], "actor_loss", "Actor (Policy) Loss", "tab:blue"),
        (axes[1], "critic_loss", "Critic (Q-Network) Loss", "tab:red"),
        (axes[2], "ent_coef", "Entropy Coefficient (alpha)", "tab:green"),
        (axes[3], "ent_coef_loss", "Entropy Coef. Loss", "tab:orange"),
    ]:
        raw = df[col].astype(float)
        smoothed = raw.rolling(window=window_size, min_periods=1).mean()
        ax.plot(df["timestep"], raw, alpha=0.2, color=color)
        ax.plot(df["timestep"], smoothed, color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("SAC Training Losses", fontsize=14)
    plt.tight_layout()

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
    csv_path = os.path.join(results_dir, "loss_log.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"No loss log found at {csv_path}")
        return

    df = df.dropna(subset=["policy_gradient_loss", "value_loss"])
    if df.empty:
        print("Loss log has no valid entries yet.")
        return

    metrics = [
        ("policy_gradient_loss", "Policy Gradient Loss", "tab:blue"),
        ("value_loss", "Value Loss", "tab:red"),
        ("entropy_loss", "Entropy Loss", "tab:green"),
        ("approx_kl", "Approx KL Divergence", "tab:orange"),
        ("clip_fraction", "Clip Fraction", "tab:purple"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.5 * len(metrics)), sharex=True)

    for ax, (col, label, color) in zip(axes, metrics):
        raw = df[col].astype(float)
        smoothed = raw.rolling(window=window_size, min_periods=1).mean()
        ax.plot(df["timestep"], raw, alpha=0.2, color=color)
        ax.plot(df["timestep"], smoothed, color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("PPO Training Losses", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"PPO loss curves saved to {plot_path}")