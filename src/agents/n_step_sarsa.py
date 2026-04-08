from .agent import Agent
import numpy as np

class NStepSARSA(Agent):
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=1000, n_step=3, seed=None):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)
        self.n_step = n_step
        self.Q = np.zeros((2*self.n - 1, 2*self.n - 1, env.action_space.n))

    def train(self):
        step_tracker = []
        success_tracker = []

        for episode in range(self.num_episodes):
            (x, y), _ = self.env.reset()
            ix, iy = super().state_to_index((x, y))
            action = self.select_action((x, y))

            # Buffers
            states = [(x, y)]
            actions = [action]
            rewards = [0]       # Placeholder for reward at time t=0

            T = float('inf')
            t = 0
            nb_steps = 0
            success = 0

            while True:

                if t < T:
                    (x_new, y_new), reward, terminated, truncated, _ = self.env.step(action)
                    ix_new, iy_new = super().state_to_index((x_new, y_new))
                    done = terminated or truncated

                    states.append((x_new, y_new))
                    rewards.append(reward)

                    if done:
                        T = t + 1
                        if terminated:
                            success = 1
                    else:
                        next_action = self.select_action((x_new, y_new))
                        actions.append(next_action)
                        action = next_action

                tau = t - self.n_step + 1
                
                if tau >= 0:
                    G = 0.0

                    # Compute n-step return
                    for i in range(tau + 1, min(tau + self.n_step, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]

                    # Bootstrap if episode is not finished
                    if tau + self.n_step < T:
                        ix_boot, iy_boot = super().state_to_index(states[tau + self.n_step])
                        G += (self.gamma ** self.n_step) * self.Q[ix_boot, iy_boot, actions[tau + self.n_step]]

                    # Update Q-value
                    ix_tau, iy_tau = super().state_to_index(states[tau])
                    self.Q[ix_tau, iy_tau, actions[tau]] += self.alpha * (G - self.Q[ix_tau, iy_tau, actions[tau]])
                
                if tau == T - 1:
                    break

                t += 1
                nb_steps += 1

            step_tracker.append(nb_steps)
            success_tracker.append(success)
            self.epsilon = max(0.01, self.epsilon * 0.995)  # Decay epsilon
            if episode % 1 == 0:
                print(f"Episode {episode}, Steps: {nb_steps}, Epsilon: {self.epsilon:.4f}")

        return step_tracker, success_tracker

    def select_action(self, state):
        return super().epsilon_greedy(self.Q, state)

    def get_q_table(self):
        return self.Q

