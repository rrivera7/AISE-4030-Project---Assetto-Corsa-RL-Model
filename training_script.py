import torch
from environment import create_env

def main():
    # 1. Initialize the environment
    print("Initializing Assetto Corsa environment...")
    env = create_env(config_path='./config.yaml')

    # 2. Obtain State and Action Spaces
    # Observation space (State)
    obs_space = env.observation_space
    # Action space
    act_space = env.action_space

    # 3. Determine the Device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Display Environment Metadata
    print("--- Environment Ready ---")
    print(f"State Space:  {obs_space}")
    print(f"Action Space: {act_space}")
    print(f"Device:       {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
    print("-------------------------")

    # 4. Confirm Environment Step
    print("\nTesting environment step...")
    initial_state = env.reset()
    
    # Take a dummy action (sampling from the action space)
    random_action = env.action_space.sample()
    
    # Step the environment
    # Note: Modern Gym versions return 5 values (obs, reward, terminated, truncated, info)
    step_result = env.step(random_action)
    
    if step_result:
        print("Success: Environment step confirmed.")
        print(f"Sample Observation Shape: {step_result[0].shape if hasattr(step_result[0], 'shape') else 'N/A'}")
    else:
        print("Error: Environment failed to step.")

    # Cleanup
    # env.close() 

if __name__ == "__main__":
    main()