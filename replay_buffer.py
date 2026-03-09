class ReplayBufferConfigurator:
    """
    A utility class to configure or customize SB3's internal ReplayBuffer.
    """
    def __init__(self, buffer_size):
        """
        Initializes the buffer configuration. Note: SB3 instantiates the actual 
        buffer automatically inside the SAC model (from what I can tell), but this class can be used 
        to manipulate the `optimize_memory_usage` flags or custom buffer classes maybe?.
        
        Args:
            buffer_size (int): The maximum number of transitions to store.
            
        Returns:
            None
        """
        pass
