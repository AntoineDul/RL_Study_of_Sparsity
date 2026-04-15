import numpy as np

from .agent import Agent

class Sarsa(Agent):
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=1000, seed=None):
        super().__init__(env, alpha, gamma, epsilon, num_episodes, seed)

        # Initialize Q-table with zeros
        self.Q = np.zeros((2*self.n - 1, 2*self.n - 1, env.action_space.n))
        

    def train(self):

        step_tracker = []
        success_tracker = []
        returns = []
        first_success_episodes = []

        for episode in range(self.num_episodes):
            # Reset the environment and get the initial state, (x,y) is the state
            (x, y), _ = self.env.reset()
            ix, iy = super().state_to_index((x, y)) # Convert state to indices for Q-table

            action = self.select_action((x, y))
            done = False
            first_success_episode = None
            nb_steps = 0
            success = 0
            total_reward = 0.0

            while not done and nb_steps < self.env.max_steps:
                (x_new, y_new), reward, terminated, truncated, _ = self.env.step(action)
                ix_new, iy_new = super().state_to_index((x_new, y_new)) # Convert new state to indices for Q-table
                done = terminated or truncated
                
                next_action = self.select_action((x_new, y_new))

                if done:
                    target = reward
                    if terminated:
                        success = 1
                        if first_success_episode is None:
                            first_success_episode = episode
                else:
                    target = reward + self.gamma * self.Q[ix_new, iy_new, next_action]

                self.Q[ix, iy, action] += self.alpha * (target - self.Q[ix, iy, action])

                (x, y) = (x_new, y_new)
                action = next_action
                total_reward += reward
                nb_steps += 1

            # Track performance metrics
            step_tracker.append(nb_steps)
            success_tracker.append(success)
            returns.append(total_reward)
            first_success_episodes.append(first_success_episode)

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995) 

            if episode % 50 == 0:
                print(f"Episode {episode}, Steps: {nb_steps}, Epsilon: {self.epsilon:.4f}")

        history = {
            "returns": returns,
            "successes": success_tracker,
            "episode_lengths": step_tracker,
            "first_success_episode": first_success_episode
        }

        return step_tracker, success_tracker

# --- Helper functions ---

    def select_action(self, state):
        return super().epsilon_greedy(self.Q, state)
    
    def get_q_table(self):
        return self.Q