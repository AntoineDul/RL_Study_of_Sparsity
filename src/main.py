from environment import GridWorldEnv
from sarsa import Sarsa
from plot import plot_step_tracker, plot_single_run

from agents.q_learning import QLearning
from agents.q_learning_bonus import QLearningBonus

from evaluation import run_agent, aggregate_histories, print_summary_table
from plotting import plot_comparison

def main():
    env_kwargs = {
        "size": 5,
        "reward_probability": 0.0,
    }

    common_agent_kwargs = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.05,
        "num_episodes": 1000,
    }

    q_histories = run_agent(
        agent_class=QLearning,
        env_class=GridWorldEnv,
        env_kwargs=env_kwargs,
        agent_kwargs=common_agent_kwargs,
        n_seeds=3,
        label="Q-Learning"
    )
    print("Q-Learning training completed\n")

    qb_histories = run_agent(
        agent_class=QLearningBonus,
        env_class=GridWorldEnv,
        env_kwargs=env_kwargs,
        agent_kwargs={**common_agent_kwargs, "beta": 1.0},
        n_seeds=3,
        label="Q-Learning + Bonus"
    )
    print("Q-Learning + Bonus training completed\n")

    results = {
        "Q-Learning": aggregate_histories(q_histories),
        "Q-Learning + Bonus": aggregate_histories(qb_histories),
    }

    print_summary_table(results)

    plot_comparison(
        results,
        metric_key="successes_mean",
        title="Success Rate Comparison",
        ylabel="Success Rate",
        window=50,
        save_path="success_comparison.png"
    )
    print("    Saved: success_comparison.png\n")

    # plot_comparison(
    #     results,
    #     metric_key="returns_mean",
    #     title="Return Comparison",
    #     ylabel="Return",
    #     window=50,
    #     save_path="return_comparison.png"
    # )
    # print("    Saved: return_comparison.png\n")

    plot_comparison(
        results,
        metric_key="lengths_mean",
        title="Episode Length Comparison",
        ylabel="Steps",
        window=50,
        save_path="length_comparison.png"
    )
    print("    Saved: length_comparison.png\n")


# def main():

#     # Initialize seed for reproducibility
#     seed = 42

#     # Initialize environment
#     env = GridWorldEnv(size=30, reward_probability=0.5, seed=seed)
    
#     # Initialize and train agent
#     agent = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.5, num_episodes=1500)
#     step_tracker, success_tracker = agent.train()

#     # Plot results (steps to goal over episodes)
#     # plot_step_tracker(step_tracker)
#     plot_single_run(success_tracker, None, window=50, save_path="plot_single_run.png")


# def test_env():
#     env = GridWorldEnv(size=5)
#     obs, _ = env.reset()
#     done = False

#     while not done:
#         action = env.action_space.sample()  # Random action
#         obs, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         env.render()

if __name__ == "__main__":
    main()