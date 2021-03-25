import numpy as np
import random
random.seed(12345)

class ReplayBuffer():
    def __init__(self, buffer_size=5000):
        self.buffer=[]
        self.buffer_size=buffer_size
        self.exp_size = 0

    def add(self, experience):
        self.exp_size = experience.shape[-1]
        if len(self.buffer)+len(experience) >= self.buffer_size:
            self.buffer[0:(len(self.buffer)+len(experience))-self.buffer_size]=[]

        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,self.exp_size])

    def get_buffer(self):
        return np.reshape(np.array(self.buffer),[len(self.buffer),self.exp_size])

    def get_newest(self, num=1):
        num = min(num, len(self.buffer))
        return np.reshape(np.array(self.buffer[-num:]), [num, self.exp_size])

    def clear_buffer(self):
        self.buffer = []
        self.exp_size = 0
