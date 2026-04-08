from collections import defaultdict
import numpy as np

from .agent import Agent


class QLearning(Agent):
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.05,
        num_episodes=3000,
        seed=None
    ):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)

        self.n_actions = self.env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def select_action(self, state):
        state = tuple(state)
        return self.epsilon_greedy(self.q_table[state])

    def train(self):
        returns = []
        successes = []
        episode_lengths = []
        first_success_episode = None

        for episode in range(self.num_episodes):
            print(f"Episode {episode+1}/{self.num_episodes}", end="\r")
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

                # Q-learning target
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * np.max(self.q_table[next_state])

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

        history = {
            "returns": returns,
            "successes": successes,
            "episode_lengths": episode_lengths,
            "first_success_episode": first_success_episode
        }

        return history

    def get_q_table(self):
        return self.q_table