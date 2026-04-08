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

        for episode in range(self.num_episodes):
            # Reset the environment and get the initial state, (x,y) is the state
            (x, y), _ = self.env.reset()
            ix, iy = super().state_to_index((x, y)) # Convert state to indices for Q-table

            action = self.select_action((x, y))
            done = False
            nb_steps = 0
            success = 0

            while not done and nb_steps < self.env.max_steps:
                (x_new, y_new), reward, terminated, truncated, _ = self.env.step(action)
                ix_new, iy_new = super().state_to_index((x_new, y_new)) # Convert new state to indices for Q-table
                done = terminated or truncated
                
                next_action = self.select_action((x_new, y_new))

                if done:
                    target = reward
                    success = 1
                else:
                    target = reward + self.gamma * self.Q[ix_new, iy_new, next_action]

                self.Q[ix, iy, action] += self.alpha * (target - self.Q[ix, iy, action])

                (x, y) = (x_new, y_new)
                action = next_action
                nb_steps += 1

            step_tracker.append(nb_steps)
            success_tracker.append(success)
            self.epsilon = max(0.01, self.epsilon * 0.995)  # Decay epsilon

            if episode % 50 == 0:
                print(f"Episode {episode}, Steps: {nb_steps}, Epsilon: {self.epsilon:.4f}")

        return step_tracker, success_tracker

# --- Helper functions ---

    # def state_to_index(self, state):
    #     dx, dy = state
    #     offset = self.n - 1
    #     return dx + offset, dy + offset

    # def epsilon_greedy(self, state):
    #     # Parse the state (observations)
    #     ix, iy = self.state_to_index(state)
        
    #     # Choose action using epsilon-greedy policy
    #     if np.random.rand() < self.epsilon:
    #         return self.env.action_space.sample()  # Explore
    #     else:
    #         return np.argmax(self.Q[ix, iy])  # Exploit

    def select_action(self, state):
        return super().epsilon_greedy(self.Q, state)
    
    def get_q_table(self):
        return self.Q