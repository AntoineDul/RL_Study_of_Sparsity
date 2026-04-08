import numpy as np


def run_agent(agent_class, env_class, env_kwargs, agent_kwargs, n_seeds=3, label="Agent"):
    histories = []

    for seed in range(n_seeds):
        env = env_class(**env_kwargs, seed=seed)
        agent = agent_class(env=env, seed=seed, **agent_kwargs)
        history = agent.train()
        histories.append(history)
        print(f"{label} | seed={seed} done")

    return histories


def aggregate_histories(histories):
    returns = np.array([h["returns"] for h in histories])
    successes = np.array([h["successes"] for h in histories])
    lengths = np.array([h["episode_lengths"] for h in histories])

    first_successes = [
        h["first_success_episode"]
        for h in histories
        if h["first_success_episode"] is not None
    ]

    return {
        "returns_mean": returns.mean(axis=0),
        "returns_std": returns.std(axis=0),
        "successes_mean": successes.mean(axis=0),
        "successes_std": successes.std(axis=0),
        "lengths_mean": lengths.mean(axis=0),
        "lengths_std": lengths.std(axis=0),
        "first_success_mean": np.mean(first_successes) if first_successes else None
    }


def print_summary_table(results):
    print("\n===== Final Summary =====")
    for label, summary in results.items():
        final_success = summary["successes_mean"][-100:].mean()
        final_return = summary["returns_mean"][-100:].mean()
        final_length = summary["lengths_mean"][-100:].mean()
        first_success = summary["first_success_mean"]

        print(f"\n{label}")
        print(f"  Avg success rate (last 100 eps): {final_success:.3f}")
        print(f"  Avg return (last 100 eps):       {final_return:.3f}")
        print(f"  Avg episode length (last 100):   {final_length:.3f}")
        print(f"  Avg first success episode:       {first_success}")