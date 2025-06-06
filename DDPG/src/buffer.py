import random
from collections import deque
import numpy as np
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=int(buffer_size))  
        self.batch_size = batch_size

    def append(self, state, action, reward, next_state, done):
        self.buffer.append([
            state, action, np.expand_dims(reward, -1),
            next_state, np.expand_dims(done, -1)
        ])
    def sample(self):

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer
