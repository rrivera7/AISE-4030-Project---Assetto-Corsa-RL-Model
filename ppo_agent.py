import os
import json
from stable_baselines3 import PPO
from actor_critic import CustomTelemetryExtractor


class PPOAgent:
    """
    Wrapper around the Stable-Baselines3 PPO implementation.
    Reads all hyperparameters from the project config.yaml so that
    experiments are fully reproducible from a single configuration file.

    Args:
        env (gym.Env): The wrapped Assetto Corsa Gym environment.
        config (OmegaConf): The full project configuration object.
    """

    def __init__(self, env, config):
        """
        Initializes the PPO model with hyperparameters from config.yaml.

        Args:
            env (gym.Env): A Gym-compatible environment instance.
            config (OmegaConf): Loaded config with ppo, policy, train sections.
        """
        self.env = env
        self.config = config

        policy_kwargs = dict(
            net_arch=list(config.policy.net_arch),
        )

        if config.policy.get("use_custom_extractor", False):
            policy_kwargs["features_extractor_class"] = CustomTelemetryExtractor
            policy_kwargs["features_extractor_kwargs"] = dict(
                features_dim=config.policy.features_dim
            )

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
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
        Persists the full PPO model (weights + optimizer state) to disk.

        Args:
            filepath (str): Destination path (without extension; SB3 adds .zip).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Loads a previously saved PPO model from disk, rebinding it to
        the current environment and device.

        Args:
            filepath (str): Path to the saved model file.
        """
        self.model = PPO.load(filepath, env=self.env, device=self.config.train.device)
        print(f"Model loaded from {filepath}")

    def save_checkpoint(self, path):
        """
        Saves a training checkpoint: model weights and training metadata.
        PPO is on-policy so there is no replay buffer to persist.

        Args:
            path (str): Directory to save the checkpoint into.
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model")
        meta_path = os.path.join(path, "training_meta.json")

        self.model.save(model_path)

        meta = {
            "num_timesteps": self.model.num_timesteps,
            "num_episodes": getattr(self.model, '_episode_num', 0),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """
        Loads a training checkpoint: model weights and training metadata.

        Args:
            path (str): Directory containing the checkpoint files.

        Returns:
            dict: Training metadata (num_timesteps, num_episodes).
        """
        model_path = os.path.join(path, "model")
        meta_path = os.path.join(path, "training_meta.json")

        self.model = PPO.load(model_path, env=self.env, device=self.config.train.device)
        print(f"Model loaded from {model_path}")

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
