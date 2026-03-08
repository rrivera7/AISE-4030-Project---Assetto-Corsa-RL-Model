"""
replay_buffer.py
Responsibility: Memory storage interface for off-policy transitions.
"""

class ReplayBufferConfigurator:
    """
    A utility class to configure or customize SB3's internal ReplayBuffer.
    """
    def __init__(self, buffer_size):
        """
        Initializes the buffer configuration. Note: SB3 instantiates the actual 
        buffer automatically inside the SAC model, but this class can be used 
        to manipulate the `optimize_memory_usage` flags or custom buffer classes.
        
        Args:
            buffer_size (int): The maximum number of transitions to store.
            
        Returns:
            None
        """
        pass

    def get_buffer_kwargs(self):
        """
        Returns a dictionary of keyword arguments to pass to the SB3 SAC algorithm 
        to customize its internal memory storage.
        
        Args:
            None
            
        Returns:
            dict: Configuration arguments for the replay buffer.
        """
        pass