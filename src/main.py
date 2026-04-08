from environment import GridWorldEnv
from agents.sarsa import Sarsa
import agents 
from plot import plot_step_tracker, plot_single_run

def main():

    # Initialize seed for reproducibility
    seed = 42

    # Initialize environment
    env = GridWorldEnv(size=30, reward_probability=0.0005, seed=seed) # good reward probability for 30x30 is 0.0005
    
    # Initialize and train agent
    #agent = agents.Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.5, num_episodes=1500, seed=seed)
    agent = agents.NStepSARSA(env, alpha=0.1, gamma=0.99, epsilon=0.5, num_episodes=1500, n_step=1, seed=seed)
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