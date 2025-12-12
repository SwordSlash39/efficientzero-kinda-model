import torch
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage
from tensordict import TensorDict

class PriorityReplay:
    def __init__(self, max_size, device, batch_size=256):
        self.buffer_max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.replaybuffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=1.0,
            storage=LazyTensorStorage(self.buffer_max_size, device=self.device),
            batch_size=self.batch_size
        )
        self.current_sample = None
    
    def add_single_state(self, data: dict):
        state_data = TensorDict(data, batch_size=[])
        self.replaybuffer.add(state_data)
    
    def add_states(self, data: dict, batch_size: int):
        state_data = TensorDict(data, batch_size=[batch_size])
        self.replaybuffer.add(state_data)
    
    def sample(self, batch_size=None):
        sample_batch_size = self.batch_size if batch_size is None else batch_size
        self.current_sample = self.replaybuffer.sample(batch_size=sample_batch_size)
        
        return self.current_sample
    
    def updateLossesFromSample(self, td_errors):
        if self.current_sample is None:
            raise RuntimeError("No current sample detected")
        
        self.current_sample.set("td_error", td_errors)
        self.replaybuffer.update_tensordict_priority(self.current_sample)
        self.current_sample = None