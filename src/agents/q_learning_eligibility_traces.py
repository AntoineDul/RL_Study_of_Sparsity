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

        # Same dense tabular layout as SARSA-family
        self.Q = np.zeros((2 * self.n - 1, 2 * self.n - 1, self.n_actions))

    def select_action(self, state):
        return super().epsilon_greedy(self.Q, state)

    def _is_greedy_action(self, state, action):
        """Return True if action is one of the greedy actions.

        With Q initialized to all zeros, several actions can tie for max.
        Treating only np.argmax(...) as greedy clears traces incorrectly.
        """
        ix, iy = super().state_to_index(state)
        q_values = self.Q[ix, iy]
        return np.isclose(q_values[action], np.max(q_values))

    def train(self):
        returns = []
        successes = []
        episode_lengths = []
        first_success_episode = None

        for episode in range(self.num_episodes):
            print(f"Episode {episode + 1}/{self.num_episodes}", end="\r")

            state, _ = self.env.reset()
            state = tuple(state)

            # Dense eligibility trace table. Reset every episode.
            E = np.zeros_like(self.Q)

            # Select once, then carry this action forward. Do not sample a
            # throwaway next_action just to decide whether to clear traces.
            action = self.select_action(state)

            done = False
            total_reward = 0.0
            steps = 0
            success = 0

            while not done and steps < self.env.max_steps:
                ix, iy = super().state_to_index(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                if terminated:
                    success = 1
                    if first_success_episode is None:
                        first_success_episode = episode

                if done:
                    td_target = reward
                    next_action = None
                else:
                    ix_next, iy_next = super().state_to_index(next_state)
                    td_target = reward + self.gamma * np.max(self.Q[ix_next, iy_next])
                    next_action = self.select_action(next_state)

                delta = td_target - self.Q[ix, iy, action]

                # Replacing traces are usually more stable than accumulating
                # traces here and avoid repeated large trace values.
                E[ix, iy, action] = 1.0

                self.Q += self.alpha * delta * E

                if done:
                    E.fill(0.0)
                elif self._is_greedy_action(next_state, next_action):
                    E *= self.gamma * self.lam
                else:
                    # Watkins Q(lambda): clear traces after the actual selected
                    # next action is exploratory/non-greedy.
                    E.fill(0.0)

                state = next_state
                action = next_action
                total_reward += reward
                steps += 1

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

    def get_q_table(self):
        return self.Q
