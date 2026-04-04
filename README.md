# AISE-4030 Project: Assetto Corsa Reinforcement Learning Model

## This project involves the development and implementation of an autonomous driving agent for the Assetto Corsa simulation environment. The system utilizes the Soft Actor-Critic (SAC) algorithm, a state-of-the-art reinforcement learning framework, to train a model capable of navigating complex vehicle physics. The primary objective is to optimize lap times and vehicle stability using a BMW Z4 GT3 on the Monza circuit.

# Installation and Setup
## Prerequisites
### Anaconda/Miniconda is required.
### Use a Conda environment with Python 3.9 (the gym plugin defaults to an env name `p309`, which maps to Python 3.9 in this project setup).

## Environment Setup
```bash
conda create -n p309 python=3.9 -y
conda activate p309
```

## Install the Gym Package
### From the repository root:
```bash
cd assetto_corsa_gym
pip install .
cd ..
```

## Install Core Training Dependencies
```bash
pip install torch stable-baselines3 omegaconf
```

## Run a Basic Environment Validation
```bash
python training_script.py
```

## Notes
### If your Conda env has a different name or Python executable path, update the plugin config accordingly (the gym plugin can also use an explicit `config_python_executable` path instead of the default Anaconda env lookup).

# Training, Checkpointing, and Results

## Starting Training
### Run the training script from the repository root:
```bash
python training_script.py
```
### This launches the full SAC training pipeline:
1. Loads all hyperparameters and environment settings from `config.yaml`.
2. Creates the Assetto Corsa Gym environment.
3. Instantiates the SAC agent with the configured policy network.
4. Trains for `total_timesteps` steps (default: 1,000,000).

### Key `config.yaml` training parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.total_timesteps` | `1000000` | Total environment steps to train |
| `train.device` | `cuda` | `cuda` for GPU, `cpu` for CPU |
| `train.save_path` | `./models/sac_assetto_corsa` | Directory for model checkpoints |
| `eval.eval_freq` | `50000` | Steps between checkpoint saves |
| `eval.log_loss_freq` | `100` | Steps between loss CSV writes |

## Checkpoints and Resuming Training
### The training pipeline includes full checkpoint support so training can be stopped and resumed without losing progress.

### Automatic Periodic Checkpoints
- Every `eval_freq` steps (default: 50,000), a full checkpoint is saved to `models/sac_assetto_corsa/checkpoint_latest/`, containing:
  - `model.zip` — SAC network weights and optimizer state.
  - `replay_buffer.pkl` — The full replay buffer contents.
  - `training_meta.json` — Training metadata (timestep count, episode count).

### Graceful Shutdown (Ctrl+C / SIGTERM)
- If the process is interrupted with **Ctrl+C** or receives a **SIGTERM** signal, a signal handler triggers an emergency checkpoint save before exiting.
- This means you keep all progress up to the moment of interruption, not just the last periodic checkpoint.

### Resuming From a Checkpoint
- Simply re-run `python training_script.py`. On startup, the script automatically checks for `checkpoint_latest/model.zip`.
- If found, it loads the model weights, replay buffer, and metadata, then continues training from the saved timestep count.
- If not found, it starts fresh.
- The SB3 timestep counter is preserved (`reset_num_timesteps=False`), so logged metrics and learning schedules continue seamlessly.

### Manual Evaluation of a Saved Model
```python
# In training_script.py main(), comment out train_agent() and uncomment:
evaluate_agent("./models/sac_assetto_corsa/sac_assetto_corsa_final")
```

## Logged Metrics and Results
### All training metrics are saved as CSV files in the `SAC_Results/` directory. These can be loaded for plotting or analysis without retraining.

### `SAC_Results/episode_log.csv`
Logged at the end of **every episode**:

| Column | Description |
|--------|-------------|
| `timestep` | Global environment step at episode end |
| `episode` | Running episode number |
| `episode_reward` | Cumulative (undiscounted) reward for the episode |
| `episode_length` | Number of steps in the episode |

### `SAC_Results/loss_log.csv`
Logged every `log_loss_freq` steps (default: 100):

| Column | Description |
|--------|-------------|
| `timestep` | Global environment step |
| `actor_loss` | SAC policy (actor) network loss |
| `critic_loss` | SAC Q-network (critic) loss |
| `ent_coef` | Current entropy coefficient (alpha) — the exploration parameter |
| `ent_coef_loss` | Entropy coefficient tuning loss |

### Generated Plots
At the end of training, two PNG plots are saved to `SAC_Results/`:
- **`learning_curve.png`** — Episode reward over time (raw + smoothed rolling average with std band).
- **`loss_curves.png`** — Four-subplot figure showing actor loss, critic loss, entropy coefficient, and entropy coefficient loss over timesteps.

### Loading Results for Custom Analysis
```python
import pandas as pd

episodes = pd.read_csv("SAC_Results/episode_log.csv")
losses = pd.read_csv("SAC_Results/loss_log.csv")
```

### Regenerating Plots Without Retraining
```python
from utils import plot_learning_curve, plot_loss_curves

plot_learning_curve("./SAC_Results")
plot_loss_curves("./SAC_Results")
```

# Core Components and Architecture
## Current Implementation
### training_script.py: Main entry point. Contains `train_agent()` (full training pipeline with automatic resume support and signal-handler-based emergency saves), `evaluate_agent()` (deterministic evaluation over N episodes), and `set_global_seeds()` for reproducibility.

### sac_agent.py: `SACAgent` wrapper around SB3 SAC. Provides action selection, training, model save/load, and full checkpoint save/load (model weights + replay buffer + training metadata).

### environment.py: Creates the environment by loading `config.yaml` with OmegaConf and calling `assettoCorsa.make_ac_env(...)` from `assetto_corsa_gym`.

### config.yaml: Central configuration for Assetto Corsa environment parameters, SAC hyperparameters, training settings, and evaluation settings.

### actor_critic.py: Defines `CustomTelemetryExtractor` (SB3 feature extractor class) for custom observation feature processing before policy/value networks.

### replay_buffer.py: Defines `ReplayBufferConfigurator` as a utility for replay-buffer-related configuration (SB3 manages the actual buffer internally).

### utils.py: Training utilities including `ModelCallback` (SB3 callback for periodic full checkpoints, per-step loss/entropy logging, and per-episode reward/length logging), `plot_learning_curve()`, and `plot_loss_curves()`.

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