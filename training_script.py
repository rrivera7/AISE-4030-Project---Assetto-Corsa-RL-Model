import os
import csv
import signal
import numpy as np
import torch
from omegaconf import OmegaConf

from environment import create_env
from sac_agent import SACAgent
from ppo_agent import PPOAgent
from utils import (ModelCallback, PPOModelCallback,
                   plot_learning_curve, plot_loss_curves, plot_ppo_loss_curves)

# Global reference so signal handlers can trigger a save
_active_agent = None
_active_save_path = None


def set_global_seeds(seed):
    """
    Sets random seeds for reproducibility across numpy, torch, and CUDA.

    Args:
        seed (int): The seed value from config.yaml.

    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _emergency_save(signum, frame):
    """
    Signal handler that saves a full checkpoint before exiting.
    Triggered by Ctrl+C (SIGINT) or SIGTERM.
    """
    global _active_agent, _active_save_path
    sig_name = signal.Signals(signum).name
    print(f"\n[{sig_name}] Interrupted — saving emergency checkpoint...")
    if _active_agent is not None and _active_save_path is not None:
        checkpoint_dir = os.path.join(_active_save_path, "checkpoint_latest")
        try:
            _active_agent.save_checkpoint(checkpoint_dir)
            print("Emergency checkpoint saved. You can resume training later.")
        except Exception as e:
            print(f"Emergency save failed: {e}")
    raise SystemExit(1)


def train_agent():
    """
    Full training pipeline with resume support:
      1. Load config
      2. Seed everything
      3. Create the Assetto Corsa environment
      4. Instantiate the SAC agent
      5. Check for an existing checkpoint and resume if found
      6. Train with checkpoint + metric logging callbacks
      7. Save final model and generate plots

    Returns:
        None
    """
    global _active_agent, _active_save_path

    print("Loading configuration...")
    config = OmegaConf.load('./config.yaml')
    set_global_seeds(config.seed)

    algo = config.get("algorithm", "sac").lower()
    log_dir = os.path.join(config.train.save_path, "logs")

    if algo == "ppo":
        results_dir = "./PPO_Results"
        AgentClass = PPOAgent
        CallbackClass = PPOModelCallback
    else:
        results_dir = "./SAC_Results"
        AgentClass = SACAgent
        CallbackClass = ModelCallback

    print("Initializing Assetto Corsa environment...")
    env = create_env(config=config, log_dir=log_dir)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    print(f"Initializing {algo.upper()} Agent...")
    agent = AgentClass(env, config)

    _active_agent = agent
    _active_save_path = config.train.save_path
    signal.signal(signal.SIGINT, _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    callback = CallbackClass(
        check_freq=config.eval.eval_freq,
        log_loss_freq=config.eval.log_loss_freq,
        save_path=config.train.save_path,
        results_dir=results_dir,
        verbose=1,
    )

    checkpoint_dir = os.path.join(config.train.save_path, "checkpoint_latest")
    resuming = os.path.exists(os.path.join(checkpoint_dir, "model.zip"))

    if resuming:
        print("--- Resuming from checkpoint ---")
        meta = agent.load_checkpoint(checkpoint_dir)
        if "num_episodes" in meta:
            callback._episode_count = meta["num_episodes"]
        completed = meta.get("num_timesteps", 0)
        remaining = config.train.total_timesteps - completed
        if remaining <= 0:
            print(f"Training already complete ({completed}/{config.train.total_timesteps} steps). Skipping.")
        else:
            print(f"Resuming training: {completed} steps done, {remaining} remaining.")
            agent.resume_train(
                total_timesteps=config.train.total_timesteps,
                callback=callback,
                log_interval=config.train.log_interval,
            )
    else:
        print("--- Starting Training Loop ---")
        agent.train(
            total_timesteps=config.train.total_timesteps,
            callback=callback,
            log_interval=config.train.log_interval,
        )

    final_path = os.path.join(config.train.save_path, f"{algo}_assetto_corsa_final")
    agent.save_model(final_path)

    print("Generating plots...")
    plot_learning_curve(results_dir, title=f"{algo.upper()} Learning Curve")
    if algo == "ppo":
        plot_ppo_loss_curves(results_dir)
    else:
        plot_loss_curves(results_dir)

    env.close()
    print("Training complete.")


def evaluate_agent(model_path, n_episodes=10):
    """
    Loads a trained model and runs deterministic evaluation episodes.
    Prints per-episode and aggregate reward statistics.

    Args:
        model_path (str): Path to the saved SB3 model .zip file.
        n_episodes (int): Number of full episodes to evaluate.

    Returns:
        None
    """
    config = OmegaConf.load('./config.yaml')
    set_global_seeds(config.seed)

    algo = config.get("algorithm", "sac").lower()

    env = create_env(config=config)
    if algo == "ppo":
        agent = PPOAgent(env, config)
        results_dir = "./PPO_Results"
    else:
        agent = SACAgent(env, config)
        results_dir = "./SAC_Results"
    agent.load_model(model_path)

    os.makedirs(results_dir, exist_ok=True)
    eval_csv_path = os.path.join(results_dir, "eval_results.csv")

    print(f"Evaluating for {n_episodes} episodes (deterministic)...")
    episode_rows = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        episode_rows.append((ep + 1, ep_reward, ep_steps))
        print(f"  Episode {ep + 1}: reward={ep_reward:.2f}  steps={ep_steps}")

    rewards = np.array([r[1] for r in episode_rows])
    print(f"\nEvaluation results over {n_episodes} episodes:")
    print(f"  Mean reward:  {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"  Min reward:   {rewards.min():.2f}")
    print(f"  Max reward:   {rewards.max():.2f}")

    with open(eval_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps"])
        writer.writerows(episode_rows)
        writer.writerow([])
        writer.writerow(["mean_reward", "std_reward", "min_reward", "max_reward"])
        writer.writerow([f"{rewards.mean():.4f}", f"{rewards.std():.4f}",
                         f"{rewards.min():.4f}", f"{rewards.max():.4f}"])
    print(f"Evaluation results saved to {eval_csv_path}")

    env.close()


def main():
    """
    Entry point. Reads 'mode' from config.yaml:
      - "train"    -> runs the training pipeline
      - "evaluate" -> loads the model at 'evaluate_model_path' and evaluates

    Returns:
        None
    """
    config = OmegaConf.load('./config.yaml')
    mode = config.get("mode", "train")

    if mode == "evaluate":
        model_path = config.get("evaluate_model_path")
        n_episodes = config.eval.n_eval_episodes
        evaluate_agent(model_path, n_episodes=n_episodes)
    else:
        train_agent()


if __name__ == "__main__":
    main()