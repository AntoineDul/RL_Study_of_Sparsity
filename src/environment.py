import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, size=5):
        super().__init__()
        self.size = size

        # Actions
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int)

        # Initialize agent and goal positions
        self.goal = None
        self.agent_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
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

        terminated = np.array_equal(self.agent_pos, self.goal)
        truncated = False

        # Only reward when the goal is reached -> sparse reward
        reward = 1 if terminated else 0
        
        # Observation is the relative position to the goal
        agent_x, agent_y = self.agent_pos
        goal_x, goal_y = self.goal
        obs = [goal_x - agent_x, goal_y - agent_y]

        return obs, reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.size, self.size), ".")
        grid[self.goal[0], self.goal[1]] = "G"
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()
