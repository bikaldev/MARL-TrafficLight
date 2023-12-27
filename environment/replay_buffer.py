import random

class ReplayBuffer:
    def __init__(self, max_size = 50):
        self.max_size = max_size
        self.memory = list()

    def push(self, transition):
        if(len(self.memory) < self.max_size):
            self.memory.append(transition)
            return transition
        else:
            self.memory.pop(0)
            self.memory.append(transition)
            return transition
    
    def sample(self, batch_size):
        if(batch_size > len(self.memory)):
            return None
        
        idx_limit = len(self.memory) - batch_size - 1
        idx = random.randint(0,idx_limit)
        return self.memory[idx:idx+batch_size]
    
