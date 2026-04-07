import numpy as np
from environment import GridWorldEnv

class Sarsa():
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=1000):
        self.env = env
        self.n = env.size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        # Initialize Q-table with zeros
        self.Q = np.zeros((2*self.n - 1, 2*self.n - 1, env.action_space.n))
        
    def epsilon_greedy(self, state):
        # Parse the state (observations)
        ix, iy = self.state_to_index(state)
        
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.Q[ix, iy])  # Exploit

    def train(self):

        step_tracker = []
        success_tracker = []

        for episode in range(self.num_episodes):
            # Reset the environment and get the initial state, (x,y) is the state
            (x, y), _ = self.env.reset()
            ix, iy = self.state_to_index((x, y)) # Convert state to indices for Q-table

            action = self.epsilon_greedy((x, y))
            done = False
            nb_steps = 0
            success = 0

            while not done and nb_steps < self.env.max_steps:
                (x_new, y_new), reward, terminated, truncated, _ = self.env.step(action)
                ix_new, iy_new = self.state_to_index((x_new, y_new)) # Convert new state to indices for Q-table
                done = terminated or truncated
                
                next_action = self.epsilon_greedy((x_new, y_new))

                if done:
                    target = reward
                    success = 1
                else:
                    target = reward + self.gamma * self.Q[ix_new, iy_new, next_action]

                self.Q[ix, iy, action] += self.alpha * (target - self.Q[ix, iy, action])


                (x, y) = (x_new, y_new)
                action = next_action
                nb_steps += 1

                # if nb_steps > 10000:
                #     self.env.render()

                # if nb_steps > 10100:
                #     print("Too many steps, breaking out of episode.")
                #     return step_tracker

            step_tracker.append(nb_steps)
            success_tracker.append(success)
            self.epsilon = max(0.01, self.epsilon * 0.995)  # Decay epsilon

            if episode % 1 == 0:
                print(f"Episode {episode}, Steps: {nb_steps}, Epsilon: {self.epsilon:.4f}")

        return step_tracker, success_tracker


# --- Helper functions ---

    def state_to_index(self, state):
        dx, dy = state
        offset = self.n - 1
        return dx + offset, dy + offset