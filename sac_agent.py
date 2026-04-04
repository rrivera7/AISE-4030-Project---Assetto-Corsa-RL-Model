import os
import json
from stable_baselines3 import SAC
from actor_critic import CustomTelemetryExtractor


class SACAgent:
    """
    Wrapper around the Stable-Baselines3 SAC implementation.
    Reads all hyperparameters from the project config.yaml so that
    experiments are fully reproducible from a single configuration file.

    Args:
        env (gym.Env): The wrapped Assetto Corsa Gym environment.
        config (OmegaConf): The full project configuration object.
    """

    def __init__(self, env, config):
        """
        Initializes the SAC model with hyperparameters from config.yaml.

        Args:
            env (gym.Env): A Gym-compatible environment instance.
            config (OmegaConf): Loaded config with sac, policy, train sections.
        """
        self.env = env
        self.config = config

        policy_kwargs = dict(
            net_arch=list(config.policy.net_arch),
            use_sde=config.policy.use_sde,
        )

        if config.policy.get("use_custom_extractor", False):
            policy_kwargs["features_extractor_class"] = CustomTelemetryExtractor
            policy_kwargs["features_extractor_kwargs"] = dict(
                features_dim=config.policy.features_dim
            )

        # Instantiate the SB3 SAC model with every tuneable parameter
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=config.sac.learning_rate,
            buffer_size=config.sac.buffer_size,
            batch_size=config.sac.batch_size,
            tau=config.sac.tau,
            gamma=config.sac.gamma,
            ent_coef=config.sac.ent_coef,
            target_entropy=config.sac.target_entropy,
            target_update_interval=config.sac.target_update_interval,
            train_freq=config.sac.train_freq,
            gradient_steps=config.sac.gradient_steps,
            learning_starts=config.sac.learning_starts,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config.train.device,
            seed=config.seed,
        )

    def select_action(self, state, evaluate=False):
        """
        Selects an action for the given state.

        Args:
            state (np.ndarray): Current observation from the environment.
            evaluate (bool): If True, uses deterministic (exploit-only) actions.

        Returns:
            np.ndarray: The chosen action vector [steer, throttle, brake].
        """
        action, _states = self.model.predict(state, deterministic=evaluate)
        return action

    def train(self, total_timesteps, callback=None, log_interval=10):
        """
        Runs the SB3 learning loop.

        Args:
            total_timesteps (int): Total environment steps to train for.
            callback (BaseCallback or list): SB3 callbacks for logging/checkpointing.
            log_interval (int): How often (in episodes) SB3 prints progress.
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )

    def save_model(self, filepath):
        """
        Persists the full SAC model (weights + optimizer state) to disk.

        Args:
            filepath (str): Destination path (without extension; SB3 adds .zip).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Loads a previously saved SAC model from disk, rebinding it to
        the current environment and device.

        Args:
            filepath (str): Path to the saved model file.
        """
        self.model = SAC.load(filepath, env=self.env, device=self.config.train.device)
        print(f"Model loaded from {filepath}")

    def save_checkpoint(self, path):
        """
        Saves a full training checkpoint: model weights, replay buffer,
        and training metadata so training can be resumed exactly.

        Args:
            path (str): Directory to save the checkpoint into.
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model")
        buffer_path = os.path.join(path, "replay_buffer")
        meta_path = os.path.join(path, "training_meta.json")

        self.model.save(model_path)
        self.model.save_replay_buffer(buffer_path)

        meta = {
            "num_timesteps": self.model.num_timesteps,
            "num_episodes": self.model._episode_num,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        print(f"Full checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """
        Loads a full training checkpoint: model weights, replay buffer,
        and training metadata.

        Args:
            path (str): Directory containing the checkpoint files.

        Returns:
            dict: Training metadata (num_timesteps, num_episodes).
        """
        model_path = os.path.join(path, "model")
        buffer_path = os.path.join(path, "replay_buffer.pkl")
        meta_path = os.path.join(path, "training_meta.json")

        self.model = SAC.load(model_path, env=self.env, device=self.config.train.device)
        print(f"Model loaded from {model_path}")

        if os.path.exists(buffer_path):
            self.model.load_replay_buffer(buffer_path)
            print(f"Replay buffer loaded from {buffer_path}")
        else:
            print("No replay buffer found, starting with empty buffer.")

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"Training metadata loaded: {meta}")

        return meta

    def resume_train(self, total_timesteps, callback=None, log_interval=10):
        """
        Resumes SB3 learning without resetting the timestep counter.

        Args:
            total_timesteps (int): Total environment steps to train for.
            callback (BaseCallback or list): SB3 callbacks.
            log_interval (int): How often SB3 prints progress.
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            reset_num_timesteps=False,
        )