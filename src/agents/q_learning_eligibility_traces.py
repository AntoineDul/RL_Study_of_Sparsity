from collections import defaultdict
import numpy as np

from .agent import Agent


class QLambda(Agent):

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.05,
        num_episodes=3000,
        lam=0.9,
        seed=None
    ):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)

        self.lam = lam
        self.n_actions = self.env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

    def select_action(self, state):
 
        state = tuple(state)

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def _is_greedy_action(self, state, action):
        state = tuple(state)
        greedy_action = int(np.argmax(self.q_table[state]))
        return action == greedy_action

    def train(self):
        returns = []
        successes = []
        episode_lengths = []
        first_success_episode = None

        for episode in range(self.num_episodes):
            print(f"Episode {episode + 1}/{self.num_episodes}", end="\r")

            state, _ = self.env.reset()
            state = tuple(state)

            # eligibility traces: state -> action trace vector
            traces = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

            done = False
            total_reward = 0.0
            steps = 0
            success = 0

            while not done and steps < 500:
                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                if terminated:
                    success = 1

                # TD error
                q_sa = self.q_table[state][action]

                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * np.max(self.q_table[next_state])

                delta = td_target - q_sa

                # accumulating trace
                traces[state][action] += 1.0

                # update all traced state-action pairs
                for traced_state in list(traces.keys()):
                    self.q_table[traced_state] += self.alpha * delta * traces[traced_state]

                if done:
                    traces.clear()
                else:
                    # if next action is greedy, decay traces
                    next_action = self.select_action(next_state)

                    if self._is_greedy_action(next_state, next_action):
                        for traced_state in list(traces.keys()):
                            traces[traced_state] *= self.gamma * self.lam
                    else:
                        traces.clear()

                state = next_state
                total_reward += reward
                steps += 1

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
