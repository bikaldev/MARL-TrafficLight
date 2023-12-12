class ReplayBuffer:
    def __init__(self, max_size = 50):
        self.max_size = max_size
        self.memory = list()
        