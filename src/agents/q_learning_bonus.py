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
        beta=0.05,
        seed=None
    ):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)

        self.beta = beta

        # Same dense tabular layout as QLearning / SARSA:
        # [dx_index, dy_index, action]
        self.visit_counts = np.zeros_like(self.Q, dtype=np.int32)

    def _exploration_bonus(self, ix, iy, action):
        count = self.visit_counts[ix, iy, action]
        return self.beta / np.sqrt(count)

    def train(self):
        returns = []
        successes = []
        episode_lengths = []
        first_success_episode = None

        for episode in range(self.num_episodes):
            print(f"Episode {episode + 1}/{self.num_episodes}", end="\r")

            state, _ = self.env.reset()
            state = tuple(state)

            done = False
            total_reward = 0.0
            steps = 0
            success = 0

            while not done and steps < self.env.max_steps:
                ix, iy = super().state_to_index(state)
                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                # update count
                self.visit_counts[ix, iy, action] += 1
                bonus = self._exploration_bonus(ix, iy, action)
                shaped_reward = reward + bonus

                # Q-learning target with shaped reward
                if done:
                    td_target = shaped_reward
                else:
                    ix_next, iy_next = super().state_to_index(next_state)
                    td_target = shaped_reward + self.gamma * np.max(self.Q[ix_next, iy_next])

                td_error = td_target - self.Q[ix, iy, action]
                self.Q[ix, iy, action] += self.alpha * td_error

                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = 1
                    if first_success_episode is None:
                        first_success_episode = episode

            returns.append(total_reward)
            successes.append(success)
            episode_lengths.append(steps)

        history = {
            "returns": returns,
            "successes": successes,
            "episode_lengths": episode_lengths,
            "first_success_episode": first_success_episode
        }

        return history
