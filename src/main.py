from environment import GridWorldEnv
from sarsa import Sarsa
from plot import plot_step_tracker, plot_single_run

def main():

    # Initialize seed for reproducibility
    seed = 42

    # Initialize environment
    env = GridWorldEnv(size=30, reward_probability=0.5, seed=seed)
    
    # Initialize and train agent
    agent = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.5, num_episodes=1500)
    step_tracker, success_tracker = agent.train()

    # Plot results (steps to goal over episodes)
    # plot_step_tracker(step_tracker)
    plot_single_run(success_tracker, None, window=50, save_path="plot_single_run.png")


def test_env():
    env = GridWorldEnv(size=5)
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

if __name__ == "__main__":
    main()