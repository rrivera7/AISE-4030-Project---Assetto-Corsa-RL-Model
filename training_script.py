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
    # Seed numpy's random number generator
    np.random.seed(seed)
    # Seed PyTorch's CPU random number generator
    torch.manual_seed(seed)
    # If a GPU is available, seed all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _emergency_save(signum, frame):
    """
    Signal handler that saves a full checkpoint before exiting.
    Triggered by Ctrl+C (SIGINT) or SIGTERM.
    """
    global _active_agent, _active_save_path
    # Retrieve the name of the received signal
    sig_name = signal.Signals(signum).name
    print(f"\n[{sig_name}] Interrupted — saving emergency checkpoint...")
    
    # Check if we have an active agent and a valid save path
    if _active_agent is not None and _active_save_path is not None:
        # Define the directory for the emergency checkpoint
        checkpoint_dir = os.path.join(_active_save_path, "checkpoint_latest")
        try:
            # Attempt to save the checkpoint
            _active_agent.save_checkpoint(checkpoint_dir)
            print("Emergency checkpoint saved. You can resume training later.")
        except Exception as e:
            # Catch and log any errors during the save process
            print(f"Emergency save failed: {e}")
    
    # Exit the program with a non-zero status code
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
    # Load the YAML configuration file
    config = OmegaConf.load('./config.yaml')
    # Apply the random seed specified in the config
    set_global_seeds(config.seed)

    # Determine which algorithm to use (e.g., 'sac' or 'ppo')
    algo = config.get("algorithm", "sac").lower()
    # Define the directory for TensorBoard/CSV logs
    log_dir = os.path.join(config.train.save_path, "logs")

    # Configure paths and classes based on the selected algorithm
    if algo == "ppo":
        results_dir = "./PPO_Results"
        AgentClass = PPOAgent
        CallbackClass = PPOModelCallback
    else:
        results_dir = "./SAC_Results"
        AgentClass = SACAgent
        CallbackClass = ModelCallback

    print("Initializing Assetto Corsa environment...")
    # Create the wrapped Gym environment
    env = create_env(config=config, log_dir=log_dir)

    # Print the observation and action space specifications
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    print(f"Initializing {algo.upper()} Agent...")
    # Instantiate the RL agent
    agent = AgentClass(env, config)

    # Store global references for the emergency save handler
    _active_agent = agent
    _active_save_path = config.train.save_path
    
    # Register signal handlers for graceful interruption
    signal.signal(signal.SIGINT, _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    # Initialize the custom callback for logging and checkpointing
    callback = CallbackClass(
        check_freq=config.eval.eval_freq,
        log_loss_freq=config.eval.log_loss_freq,
        save_path=config.train.save_path,
        results_dir=results_dir,
        verbose=1,
    )

    # Determine the directory to look for an existing checkpoint
    resume_sub = config.train.get("resume_checkpoint") or ""
    if resume_sub:
        checkpoint_dir = os.path.join(config.train.save_path, resume_sub)
    else:
        checkpoint_dir = os.path.join(config.train.save_path, "checkpoint_latest")
        
    # Check if a saved model zip file exists in the checkpoint directory
    resuming = os.path.exists(os.path.join(checkpoint_dir, "model.zip"))

    if resuming:
        print("--- Resuming from checkpoint ---")
        # Load the checkpoint metadata and model weights
        meta = agent.load_checkpoint(checkpoint_dir)
        
        # Restore the episode count if available
        if "num_episodes" in meta:
            callback._episode_count = meta["num_episodes"]
            
        # Calculate how many timesteps have already been completed
        completed = meta.get("num_timesteps", 0)
        remaining = config.train.total_timesteps - completed
        
        # Check if training is already finished
        if remaining <= 0:
            print(f"Training already complete ({completed}/{config.train.total_timesteps} steps). Skipping.")
        else:
            print(f"Resuming training: {completed} steps done, {remaining} remaining.")
            # Resume the learning process
            agent.resume_train(
                total_timesteps=config.train.total_timesteps,
                callback=callback,
                log_interval=config.train.log_interval,
            )
    else:
        print("--- Starting Training Loop ---")
        # Start training from scratch
        agent.train(
            total_timesteps=config.train.total_timesteps,
            callback=callback,
            log_interval=config.train.log_interval,
        )

    # Define the path for the final saved model
    final_path = os.path.join(config.train.save_path, f"{algo}_assetto_corsa_final")
    # Save the fully trained model
    agent.save_model(final_path)

    print("Generating plots...")
    # Plot the learning curve (reward over time)
    plot_learning_curve(results_dir, title=f"{algo.upper()} Learning Curve")
    
    # Plot algorithm-specific loss curves
    if algo == "ppo":
        plot_ppo_loss_curves(results_dir)
    else:
        plot_loss_curves(results_dir)

    # Close the environment to free resources
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
    # Load the configuration and set random seeds
    config = OmegaConf.load('./config.yaml')
    set_global_seeds(config.seed)

    # Determine the algorithm used
    algo = config.get("algorithm", "sac").lower()

    # Create the environment for evaluation
    env = create_env(config=config)
    
    # Instantiate the correct agent type and set the results directory
    if algo == "ppo":
        agent = PPOAgent(env, config)
        results_dir = "./PPO_Results"
    else:
        agent = SACAgent(env, config)
        results_dir = "./SAC_Results"
        
    # Load the trained model weights
    agent.load_model(model_path)

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Define the path for the evaluation CSV log
    eval_csv_path = os.path.join(results_dir, "eval_results.csv")

    print(f"Evaluating for {n_episodes} episodes (deterministic)...")
    # List to store results for each episode
    episode_rows = []

    # Run the evaluation loop for the specified number of episodes
    for ep in range(n_episodes):
        # Reset the environment at the start of the episode
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        # Step through the environment until the episode is done
        while not done:
            # Select an action deterministically
            action = agent.select_action(obs, evaluate=True)
            # Apply the action to the environment
            obs, reward, done, info = env.step(action)
            # Accumulate the reward and increment the step counter
            ep_reward += reward
            ep_steps += 1

        # Record the episode's statistics
        episode_rows.append((ep + 1, ep_reward, ep_steps))
        print(f"  Episode {ep + 1}: reward={ep_reward:.2f}  steps={ep_steps}")

    # Extract all episode rewards into a numpy array for statistical analysis
    rewards = np.array([r[1] for r in episode_rows])
    print(f"\nEvaluation results over {n_episodes} episodes:")
    print(f"  Mean reward:  {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"  Min reward:   {rewards.min():.2f}")
    print(f"  Max reward:   {rewards.max():.2f}")

    # Write the evaluation results to a CSV file
    with open(eval_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write the header row
        writer.writerow(["episode", "reward", "steps"])
        # Write the data for each episode
        writer.writerows(episode_rows)
        # Add an empty row for separation
        writer.writerow([])
        # Write the aggregate statistics headers
        writer.writerow(["mean_reward", "std_reward", "min_reward", "max_reward"])
        # Write the aggregate statistics values
        writer.writerow([f"{rewards.mean():.4f}", f"{rewards.std():.4f}",
                         f"{rewards.min():.4f}", f"{rewards.max():.4f}"])
                         
    print(f"Evaluation results saved to {eval_csv_path}")

    # Close the environment
    env.close()


def main():
    """
    Entry point. Reads 'mode' from config.yaml:
      - "train"    -> runs the training pipeline
      - "evaluate" -> loads the model at 'evaluate_model_path' and evaluates

    Returns:
        None
    """
    # Load the configuration file
    config = OmegaConf.load('./config.yaml')
    # Retrieve the execution mode, defaulting to 'train'
    mode = config.get("mode", "train")

    # Execute the appropriate pipeline based on the mode
    if mode == "evaluate":
        model_path = config.get("evaluate_model_path")
        n_episodes = config.eval.n_eval_episodes
        evaluate_agent(model_path, n_episodes=n_episodes)
    else:
        train_agent()


if __name__ == "__main__":
    main()