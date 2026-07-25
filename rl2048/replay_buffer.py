import random
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done", "next_legal_mask"],
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)
