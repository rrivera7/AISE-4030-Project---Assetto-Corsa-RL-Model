"""
training_script.py
Responsibility: Main entry point. For Task 1, this confirms the API and hardware setup.
"""
import torch
import numpy as np
from environment import create_env

def verify_environment():
    print("--- TASK 1: ENVIRONMENT SETUP & API CONFIRMATION ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Hardware] Device being used: {device}")
    
    print("\n[Init] Booting Assetto Corsa Environment...")
    env = create_env()
    
    print("[Init] Resetting environment to start state...")
    initial_state = env.reset()
    
    print("\n--- OBSERVATION (STATE) SPACE ---")
    print(f"Shape: {env.observation_space.shape}")
    print(f"Data Type: {env.observation_space.dtype}")
    print(f"Value Ranges: Min = {np.min(env.observation_space.low)}, Max = {np.max(env.observation_space.high)}")
    
    print("\n--- ACTION SPACE ---")
    is_continuous = 'Box' in str(type(env.action_space))
    print(f"Type: {'Continuous' if is_continuous else 'Discrete'}")
    print(f"Dimensions/Shape: {env.action_space.shape}")
    print(f"Value Ranges: Low = {env.action_space.low}, High = {env.action_space.high}")
    
    print("\n--- EXECUTING TEST STEP ---")
    try:
        # Using the exact action application method from the demo
        steer = 0.1
        action_array = np.array([steer, 0.1, -1.0])
        print(f"Applying action array: {action_array}")
        
        env.set_actions(action_array)
        next_state, reward, done, info = env.step(action=None)
        
        print("[Success] Confirmation that at least one environment step executed successfully!")
        print(f"Step Output -> Reward: {reward:.4f}, Done: {done}")
        
    except Exception as e:
        print(f"[Error] Environment step failed: {e}")
        
    finally:
        print("\n[Cleanup] Recovering car and closing environment...")
        if hasattr(env, 'recover_car'):
            env.recover_car()
        env.close()

if __name__ == "__main__":
    verify_environment()