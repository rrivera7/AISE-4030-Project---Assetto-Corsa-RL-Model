import torch
import numpy as np
import os

from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from environment import create_env
from utils import load_config

# ===================================================================
# --- The Training Function ---
# ===================================================================
def train(env, agent, replay_buffer, num_episodes, start_episode, batch_size,
          print_every, checkpoint_every, model_filepath, history_filepath):
    """
    Executes the main training loop for SAC using an experience replay buffer.
    """
    print(f"Starting SAC training from episode {start_episode}...")

    # Phase 2 - Task 1 Requirement: API Confirmation [cite: 187, 194, 206]
    # Execute at least one environment step successfully before the main loop [cite: 206]
    print("\n--- Running Task 1 API Confirmation Step ---")
    test_obs, _ = env.reset()
    test_action = env.action_space.sample()
    _, _, _, _, _ = env.step(test_action)
    print("Environment step executed successfully.\n")

    for episode in range(start_episode, num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        terminal = False

        while not terminal:
            # 1. Choose action (Stochastic exploration) [cite: 94]
            action = agent.choose_action(np.array(state), deterministic=False)

            # 2. Step in the environment
            next_state, reward, done, truncated, info = env.step(action)
            terminal = done or truncated

            # 3. Store transition in replay buffer (Off-policy) 
            replay_buffer.add(state, action, reward, next_state, terminal)

            # 4. Learn from a batch of memory
            if agent.total_steps > batch_size:
                losses = agent.learn(replay_buffer, batch_size)

            # 5. Move to next state
            state = next_state
            total_reward += reward

        # Logging and Checkpointing logic here...
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Steps: {agent.total_steps} | Reward: {total_reward:.2f}")

    print("Training finished.")

# ===================================================================
# --- Main Execution ---
# ===================================================================
def main():
    """
    Entry point for the SAC Autonomous Racing training script.
    """
    print("--- Starting Assetto Corsa SAC Setup ---")

    # --- 1. Load Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))

    # --- 2. Set Device (Phase 2 Task 1 Output Requirement) [cite: 205] ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device confirmed: {device}")

    # --- 3. Initialization and Space Confirmation (Phase 2 Task 1) [cite: 202, 204] ---
    env_name = cfg["environment"]["name"]
    env, state_shape, action_size = create_env(env_name)
    
    print(f"Observation Space confirmed: {env.observation_space}")
    print(f"Action Space confirmed: {env.action_space}")

    # --- 4. Agent and Buffer Setup ---
    agent_cfg = cfg["agent"]
    agent = SACAgent(
        state_dim=state_shape[0], 
        action_dim=action_size, 
        config=agent_cfg, 
        device=device
    )
    
    replay_buffer = ReplayBuffer(
        state_dim=state_shape, 
        action_dim=action_size, 
        max_size=agent_cfg["buffer_capacity"]
    )

    # Begin Training
    train(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=cfg["training"]["num_episodes"],
        start_episode=0,
        batch_size=agent_cfg["batch_size"],
        print_every=cfg["training"]["print_every"],
        checkpoint_every=cfg["training"]["checkpoint_every"],
        model_filepath=os.path.join(SCRIPT_DIR, cfg["paths"]["results_dir"], cfg["paths"]["model_filename"]),
        history_filepath=os.path.join(SCRIPT_DIR, cfg["paths"]["results_dir"], cfg["paths"]["history_filename"])
    )

    env.close()
    print("\n--- Program Finished ---")

if __name__ == "__main__":
    main()