class ReplayBufferConfigurator:
    """
    Documents and structures the replay buffer configuration used by SB3's SAC.

    Stable-Baselines3 instantiates and manages its own ReplayBuffer internally.
    This class exists to:
      1. Centralise buffer-related parameters in one place.
      2. Provide a clear extension point for advanced variants
         (e.g., Prioritized Experience Replay) without changing sac_agent.py.

    Args:
        buffer_size (int): Maximum number of transitions stored.
        optimize_memory_usage (bool): If True, SB3 stores next_obs in a
            memory-efficient way (saves ~50% RAM but limits buffer_size to
            at most ``sys.maxsize``).
    """

    def __init__(self, buffer_size: int, optimize_memory_usage: bool = False):
        # Store the maximum number of transitions the buffer can hold
        self.buffer_size = buffer_size
        # Store the flag indicating whether to optimize memory usage
        self.optimize_memory_usage = optimize_memory_usage

    def get_buffer_kwargs(self):
        """
        Returns keyword arguments that can be forwarded directly to the
        SB3 model constructor's ``replay_buffer_kwargs`` parameter.

        Returns:
            dict: Buffer configuration dictionary.
        """
        # Return a dictionary containing the memory optimization flag
        return {
            "optimize_memory_usage": self.optimize_memory_usage,
        }