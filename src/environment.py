import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, size=5, reward_probability=0.3, seed=None):
        super().__init__()
        self.size = size
        self.reward_probability = reward_probability
        self.nb_steps = 0
        self.max_steps = size * size * 8  # Arbitrary large number to prevent infinite episodes

        # Actions
        self.action_space = spaces.Discrete(4)

        # Observation space is the relative position to the goal (dx, dy)
        self.observation_space = spaces.Box(
            low=-(size - 1), 
            high= (size - 1), 
            shape=(2,), 
            dtype=np.int32
            )

        # Initialize agent and goal positions
        self.goal = None
        self.agent_pos = None
        
        # Initialize seed for reproducibility
        self.seed = seed
        np.random.seed(seed)

    def reset(self):
        super().reset(seed=self.seed)
        self.nb_steps = 0

        # Randomly initialize agent and goal positions
        self.agent_pos = np.random.randint(0, self.size, size=2)
        self.goal = np.random.randint(0, self.size, size=2)
        
        # Ensure agent and goal do not start at the same position
        while np.array_equal(self.agent_pos, self.goal):
            self.goal = np.random.randint(0, self.size, size=2)

        # Observation is the relative position to the goal
        agent_x, agent_y = self.agent_pos
        goal_x, goal_y = self.goal
        obs = [goal_x - agent_x, goal_y - agent_y]

        return obs, {}

    def step(self, action):
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal)
        x, y = self.agent_pos

        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # Left
            x = max(0, x - 1)
        elif action == 3:  # Right
            x = min(self.size - 1, x + 1)

        # New agent position
        self.agent_pos = np.array([x, y])
        new_distance_to_goal = np.linalg.norm(self.agent_pos - self.goal)


        terminated = np.array_equal(self.agent_pos, self.goal)
        truncated = False

        # Determine reward
        if new_distance_to_goal < distance_to_goal:
            reward = 0.1  # Positive reward for moving closer
        elif new_distance_to_goal > distance_to_goal:
            reward = -0.1  # Negative reward for moving away
        elif terminated:
            reward = 100  # Large reward for reaching the goal
        else:            
            reward = 0 

        if np.random.rand() >= self.reward_probability:
            reward = 0 # No reward with certain probability to increase sparsity
        
        # Observation is the relative position to the goal
        agent_x, agent_y = self.agent_pos
        goal_x, goal_y = self.goal
        obs = [goal_x - agent_x, goal_y - agent_y]

        self.nb_steps += 1

        return obs, reward, terminated, truncated, {}

# --- Helper functions ---

    def render(self):
        grid = np.full((self.size, self.size), ".")
        grid[self.goal[0], self.goal[1]] = "G"
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()
