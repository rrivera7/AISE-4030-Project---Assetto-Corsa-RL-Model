"""
training_script.py
Responsibility: Main entry point. For Task 1, this confirms the API and hardware setup.
"""
import torch
import numpy as np
from environment import create_env

def verify_environment():
    print("--- TASK 1: ENVIRONMENT SETUP & API CONFIRMATION ---")
    
    # Requirement 3: Print the device being used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Hardware] Device being used: {device}")
    
    # Initialize the environment
    print("\n[Init] Booting Assetto Corsa Environment...")
    env = create_env()
    
    # Reset to get the initial state
    print("[Init] Resetting environment to start state...")
    initial_state = env.reset()
    
    # Requirement 1: Print Observation (State) Space details
    print("\n--- OBSERVATION (STATE) SPACE ---")
    print(f"Shape: {env.observation_space.shape}")
    print(f"Data Type: {env.observation_space.dtype}")
    # Using np.min and np.max to summarize the ranges cleanly if the array is large
    print(f"Value Ranges: Min = {np.min(env.observation_space.low)}, Max = {np.max(env.observation_space.high)}")
    
    # Requirement 2: Print Action Space details
    print("\n--- ACTION SPACE ---")
    is_continuous = 'Box' in str(type(env.action_space))
    print(f"Type: {'Continuous' if is_continuous else 'Discrete'}")
    print(f"Dimensions/Shape: {env.action_space.shape}")
    print(f"Value Ranges: Low = {env.action_space.low}, High = {env.action_space.high}")
    
    # Requirement 4: Confirm one environment step executes successfully
    print("\n--- EXECUTING TEST STEP ---")
    try:
        # Sample a random continuous action
        random_action = env.action_space.sample()
        print(f"Applying random action: {random_action}")
        
        # Take the step
        next_state, reward, done, info = env.step(random_action)
        print("[Success] Confirmation that at least one environment step executed successfully!")
        print(f"Step Output -> Reward: {reward:.4f}, Done: {done}")
        
    except Exception as e:
        print(f"[Error] Environment step failed: {e}")
        
    finally:
        print("\n[Cleanup] Closing environment...")
        env.close()

if __name__ == "__main__":
    verify_environment()