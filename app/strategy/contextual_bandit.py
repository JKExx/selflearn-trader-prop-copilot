from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyOutput:
    action: int
    explore: bool
    p_up: float


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def choose(self, p_up: float, threshold: float) -> PolicyOutput:
        # Exploration
        if np.random.rand() < self.epsilon:
            a = np.random.choice([-1, 0, 1])
            return PolicyOutput(action=a, explore=True, p_up=p_up)
        # Exploitation
        if p_up > threshold:
            return PolicyOutput(action=1, explore=False, p_up=p_up)
        if (1 - p_up) > threshold:
            return PolicyOutput(action=-1, explore=False, p_up=p_up)
        return PolicyOutput(action=0, explore=False, p_up=p_up)
