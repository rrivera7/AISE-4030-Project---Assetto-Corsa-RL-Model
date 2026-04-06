# AISE-4030 Project: Assetto Corsa Reinforcement Learning Model

## Overview
This project involves the development and implementation of an autonomous driving agent for the Assetto Corsa simulation environment. The system utilizes both **Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)** algorithms to train models capable of navigating complex vehicle physics. The primary objective is to optimize lap times and vehicle stability using a BMW Z4 GT3 on the Monza circuit, and to compare the performance of off-policy (SAC) vs. on-policy (PPO) reinforcement learning methods.

---

## Installation and Setup

### Prerequisites
- Anaconda/Miniconda is required.
- Use a Conda environment with Python 3.9 (the gym plugin defaults to an env name `p309`, which maps to Python 3.9 in this project setup).

### Environment Setup
```bash
conda create -n p309 python=3.9 -y
conda activate p309
```

### Install the Gym Package
From the repository root:
```bash
cd assetto_corsa_gym
pip install .
cd ..
```

### Install Core Training Dependencies
```bash
pip install torch stable-baselines3 omegaconf pandas matplotlib
```

---

## Usage: Training and Evaluation

The main entry point is `training_script.py`, which reads parameters from `config.yaml`. 

### 1. Training an Agent
In `config.yaml`, set `mode: "train"` and choose your algorithm (`algorithm: "sac"` or `algorithm: "ppo"`).
```bash
python training_script.py
```
This launches the full training pipeline:
1. Loads hyperparameters from `config.yaml`.
2. Creates the Assetto Corsa Gym environment.
3. Instantiates the chosen agent (SAC or PPO).
4. Trains for `total_timesteps` steps.
5. Generates learning and loss curves in `SAC_Results/` or `PPO_Results/`.

### 2. Evaluating a Trained Agent (Deployment)
In `config.yaml`, set `mode: "evaluate"`, specify the `evaluate_model_path`, and set `eval.n_eval_episodes` (e.g., 50).
```bash
python training_script.py
```
**Note on Evaluation Saving:** During evaluation, the results are kept in memory and the `eval_results.csv` file is **only generated and saved after all episodes are completed**. If you stop the evaluation early (e.g., via Ctrl+C before reaching 50 episodes), the CSV file will not be created.

---

## Checkpoints and Resuming Training

The training pipeline includes full checkpoint support so training can be stopped and resumed without losing progress.

- **Periodic Checkpoints:** Saved automatically based on `eval_freq` to `checkpoint_latest/` (includes model weights, replay buffer for SAC, and metadata).
- **Episode Checkpoints:** Saved every 50 episodes (e.g., `checkpoint_ep50`, `checkpoint_ep100`).
- **Graceful Shutdown:** If the process is interrupted with **Ctrl+C** or receives a **SIGTERM** signal, an emergency checkpoint is saved before exiting.
- **Resuming:** Simply re-run `python training_script.py`. It automatically checks for `checkpoint_latest/model.zip` and resumes training from the saved timestep count.

---

## Folder Structure and Logged Metrics

### `SAC_Results/` & `PPO_Results/`
During training, metrics are saved as CSV files in their respective algorithm directories:
- **`episode_log.csv`**: Logged at the end of *every episode* (timestep, episode, episode_reward, episode_length).
- **`loss_log.csv`**: Logged periodically (actor/critic loss for SAC; policy/value/entropy loss for PPO).
- **Generated Plots**: `learning_curve.png` and `loss_curves.png` are automatically generated at the end of training.

### `Comparative_Plots/`
This folder contains comparative analysis plots evaluating SAC vs. PPO across 4 key metrics:
1. **`learning_speed.png`**: Compares how fast each algorithm reaches a target reward threshold.
2. **`loss_convergence.png`**: Compares value-function and policy loss convergence trends.
3. **`final_performance.png`**: Bar chart comparing the mean reward and standard deviation over the final N episodes.
4. **`stability_variance.png`**: Overlays smoothed reward curves with ±1 standard deviation bands to compare training stability.

---

## Core Components and Architecture

- **`config.yaml`**: Central configuration for environment parameters, hyperparameters, execution modes (`train` vs `evaluate`), and file paths.
- **`training_script.py`**: Main entry point. Contains `train_agent()` and `evaluate_agent()`.
- **`sac_agent.py` & `ppo_agent.py`**: Wrappers around SB3 algorithms. Provide action selection, training loops, and checkpoint management.
- **`environment.py`**: Environment creation, wrappers/preprocessing, and simulator hookup.
- **`actor_critic.py`**: Defines `CustomTelemetryExtractor` for custom observation feature processing.
- **`replay_buffer.py`**: Configuration utilities for the replay buffer.
- **`utils.py`**: Shared utilities including `ModelCallback`, `PPOModelCallback`, and all plotting functions for individual and comparative analysis.
