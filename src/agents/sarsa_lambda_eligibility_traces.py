import numpy as np

from .agent import Agent

class SarsaLambdaEligibilityTraces(Agent):
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        num_episodes=1000,
        lam=0.9,
        seed=None
    ):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)

        self.lam = lam

        self.Q = np.zeros((2 * self.n - 1, 2 * self.n - 1, env.action_space.n))

    def train(self):
        episode_lengths = []
        successes = []
        returns = []
        first_success_episode = None

        for episode in range(self.num_episodes):
            # reset environment
            (x, y), _ = self.env.reset()
            action = self.select_action((x, y))

            # same shape as Q
            E = np.zeros_like(self.Q)

            done = False
            nb_steps = 0
            success = 0
            total_reward = 0.0

            while not done and nb_steps < self.env.max_steps:
                ix, iy = super().state_to_index((x, y))

                # action
                (x_new, y_new), reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                if terminated:
                    success = 1
                    if first_success_episode is None:
                        first_success_episode = episode

                # TD error delta
                if done:
                    delta = reward - self.Q[ix, iy, action]
                else:
                    next_action = self.select_action((x_new, y_new))
                    ix_new, iy_new = super().state_to_index((x_new, y_new))
                    delta = reward + self.gamma * self.Q[ix_new, iy_new, next_action] - self.Q[ix, iy, action]

                # accumulating eligibility traces
                E[ix, iy, action] += 1.0

                self.Q += self.alpha * delta * E

                # decaying traces
                E *= self.gamma * self.lam

                if not done:
                    (x, y) = (x_new, y_new)
                    action = next_action

                nb_steps += 1

            episode_lengths.append(nb_steps)
            successes.append(success)
            returns.append(total_reward)

            self.epsilon = max(0.01, self.epsilon * 0.995)

            if episode % 50 == 0:
                print(
                    f"Episode {episode}, Steps: {nb_steps}, "
                    f"Success: {success}, Epsilon: {self.epsilon:.4f}"
                )

        history = {
            "returns": returns,
            "successes": successes,
            "episode_lengths": episode_lengths,
            "first_success_episode": first_success_episode
        }

        return history

    def select_action(self, state):
        return super().epsilon_greedy(self.Q, state)

    def get_q_table(self):
        return self.Q


