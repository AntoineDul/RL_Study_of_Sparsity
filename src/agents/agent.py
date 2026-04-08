from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """
    Common interface for all tabular RL agents in this project.
    Every agent should return training history in the same format so
    evaluation and plotting code can be shared.
    """

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.05,
        num_episodes=3000,
        seed=None
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def select_action(self, state):
        """
        Choose an action from the current state.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the agent and return a history dictionary with this format:

        history = {
            "returns": [...],            # total reward per episode
            "successes": [...],          # 1 if goal reached, else 0
            "episode_lengths": [...],    # steps taken in each episode
            "first_success_episode": int or None
        }
        """
        pass

    @abstractmethod
    def get_q_table(self):
        """
        Return learned Q-table or equivalent tabular value structure.
        If not applicable, return None.
        """
        pass

    def epsilon_greedy(self, q_values):
        """
        Shared helper for epsilon-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(q_values))