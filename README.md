# AISE-4030 Project: Assetto Corsa Reinforcement Learning Model

## This project involves the development and implementation of an autonomous driving agent for the Assetto Corsa simulation environment. The system utilizes the Soft Actor-Critic (SAC) algorithm, a state-of-the-art reinforcement learning framework, to train a model capable of navigating complex vehicle physics. The primary objective is to optimize lap times and vehicle stability using a BMW Z4 GT3 on the Monza circuit.

# Core Components and Architecture
## The codebase is organized into several specialized modules to handle environment interfacing, agent logic, and data processing:

## training_script.py: Serves as the main entry point for the training pipeline, handling environment instantiation, device allocation for CUDA or CPU, and testing basic environment steps.

## sac_agent.py: Defines the SACAgent class, which wraps the Stable-Baselines3 implementation to manage policy architecture, learning rates, and model serialization.

## environment.py: Facilitates the connection to the simulation using the assetto_corsa_gym library and initializes settings via OmegaConf.

## config.yaml: Acts as a centralized configuration hub for hyperparameters such as batch size, learning rate, and simulation-specific variables like sampling frequency and track selection.


## actor_critic.py: Contains a CustomTelemetryExtractor designed to process state vectors and extract relevant features before they are passed to the actor and critic networks.

## utils.py: Provides monitoring tools, including a custom callback for training management and functions for visualizing the learning curve.

## replay_buffer.py: Includes a utility class intended for the configuration or customization of the internal replay buffer used during the learning process.