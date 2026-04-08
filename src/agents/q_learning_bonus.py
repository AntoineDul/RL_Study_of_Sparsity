from collections import defaultdict
import numpy as np

from .q_learning import QLearning


class QLearningBonus(QLearning):
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.05,
        num_episodes=3000,
        beta=1.0,
        seed=None
    ):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)

        self.beta = beta
        self.visit_counts = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.int32))

    def _exploration_bonus(self, state, action):
        count = self.visit_counts[state][action]
        return self.beta / np.sqrt(count)

    def train(self):
        returns = []
        successes = []
        episode_lengths = []
        first_success_episode = None

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            state = tuple(state)

            done = False
            total_reward = 0.0
            steps = 0
            success = 0

            while not done and steps < 500:  # Prevent infinite episodes
                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                # update count first
                self.visit_counts[state][action] += 1

                bonus = self._exploration_bonus(state, action)
                shaped_reward = reward + bonus

                # SAME Q-learning update (just using shaped reward)
                if done:
                    td_target = shaped_reward
                else:
                    td_target = shaped_reward + self.gamma * np.max(self.q_table[next_state])

                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error

                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = 1

            returns.append(total_reward)
            successes.append(success)
            episode_lengths.append(steps)

            if success and first_success_episode is None:
                first_success_episode = episode

        return {
            "returns": returns,
            "successes": successes,
            "episode_lengths": episode_lengths,
            "first_success_episode": first_success_episode
        }