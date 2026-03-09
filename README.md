# AISE-4030 Project: Assetto Corsa Reinforcement Learning Model

## This project involves the development and implementation of an autonomous driving agent for the Assetto Corsa simulation environment. The system utilizes the Soft Actor-Critic (SAC) algorithm, a state-of-the-art reinforcement learning framework, to train a model capable of navigating complex vehicle physics. The primary objective is to optimize lap times and vehicle stability using a BMW Z4 GT3 on the Monza circuit.

# Core Components and Architecture
## The codebase is organized into several specialized modules to handle environment interfacing, agent logic, and data processing:

## File Explanations (Current Implementation)
### training_script.py: Main runnable script right now. It initializes the Assetto Corsa environment, reports observation/action spaces, selects CPU/GPU with PyTorch, and performs a basic reset/step validation. It also contains placeholder `train_agent()` and `evaluate_agent()` functions.

### sac_agent.py: Defines the `SACAgent` wrapper interface (constructor, action selection, train, save, load), but method bodies are currently placeholders (`pass`).

### environment.py: Creates the environment by loading `config.yaml` with OmegaConf and calling `assettoCorsa.make_ac_env(...)` from `assetto_corsa_gym`.

### config.yaml: Central configuration for Assetto Corsa environment parameters, SAC hyperparameters, training settings, and evaluation settings.

### actor_critic.py: Defines `CustomTelemetryExtractor` (SB3 feature extractor class skeleton) for custom observation feature processing before policy/value networks.

### replay_buffer.py: Defines `ReplayBufferConfigurator` as a placeholder utility for replay-buffer-related configuration ideas.

### utils.py: Shared training utility scaffolding. Currently includes `ModelCallback` (SB3 callback skeleton with `_on_step` and `load_config`) and `plot_learning_curve(...)` placeholder.

## Code Structure and Naming Conventions
### The project follows modular, single-responsibility file naming so each file has a clear role:

### `config.yaml` (fixed name): Hyperparameters, environment settings, execution modes, and file paths.
### `environment.py` (fixed name): Environment creation, wrappers/preprocessing, and simulator hookup.
### `{algorithm}_agent.py` (named after algorithm): RL agent logic (action selection, learning loop, save/load).
### `{type}_network.py` (named after network type): Neural network architecture components.
### `{type}_buffer.py` (named after buffer type): Replay/rollout memory components when applicable.
### `training_script.py` (fixed name): Main entry point and orchestration for train/eval runs.
### `utils.py` (fixed name): Shared utilities such as callbacks, plotting, logging helpers, config helpers, and related support functions.
### `README.md` (fixed name): Project overview, setup/run guide, and design documentation.