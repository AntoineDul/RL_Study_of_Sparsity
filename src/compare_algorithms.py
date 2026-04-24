import matplotlib.pyplot as plt
import numpy as np

from environment import GridWorldEnv
import agents
from utils import moving_average


# -----------------------------
# Helpers
# -----------------------------

def smooth_same_length(data, window):
    """
    Moving average padded to the same length as the input.
    Useful for episode lengths.
    """
    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        return data.copy()

    if window <= 1:
        return data.copy()

    if len(data) < window:
        return np.full(len(data), np.mean(data), dtype=float)

    smoothed = moving_average(data, window)
    pad_left = window - 1
    left_pad = np.full(pad_left, smoothed[0], dtype=float)
    return np.concatenate([left_pad, smoothed])


def cumulative_success(successes):
    """
    Cumulative success rate:
    success_rate[t] = (# successes up to episode t) / (t + 1)
    """
    successes = np.asarray(successes, dtype=float)
    if len(successes) == 0:
        return successes.copy()

    return np.cumsum(successes) / np.arange(1, len(successes) + 1)


def aggregate_runs(run_histories, metric, transform=None, smooth_window=None):
    """
    Convert a list of history dicts into:
    mean curve, std curve, x axis
    """
    curves = []

    for history in run_histories:
        y = np.asarray(history[metric], dtype=float)

        if transform is not None:
            y = transform(y)

        if smooth_window is not None:
            y = smooth_same_length(y, smooth_window)

        curves.append(y)

    min_len = min(len(curve) for curve in curves)
    curves = np.array([curve[:min_len] for curve in curves], dtype=float)

    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    x = np.arange(min_len)

    return x, mean_curve, std_curve


def plot_with_band(
    family_runs,
    metric,
    title,
    ylabel,
    save_path,
    transform=None,
    smooth_window=None,
    focus_labels=None
):
    plt.figure(figsize=(11, 7))

    items = family_runs.items()
    if focus_labels is not None:
        items = [(label, family_runs[label]) for label in focus_labels if label in family_runs]

    for label, run_histories in items:
        x, mean_curve, std_curve = aggregate_runs(
            run_histories,
            metric=metric,
            transform=transform,
            smooth_window=smooth_window,
        )

        plt.plot(x, mean_curve, linewidth=3, label=label)
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.18
        )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def print_family_summary(family_runs):
    print("\nSummary across seeds:")
    for label, run_histories in family_runs.items():
        final_success_rates = []
        final_lengths = []
        first_successes = []

        for history in run_histories:
            successes = np.asarray(history["successes"], dtype=float)
            lengths = np.asarray(history["episode_lengths"], dtype=float)

            tail_succ = successes[-50:] if len(successes) >= 50 else successes
            tail_len = lengths[-50:] if len(lengths) >= 50 else lengths

            final_success_rates.append(np.mean(tail_succ))
            final_lengths.append(np.mean(tail_len))

            first_successes.append(history.get("first_success_episode", None))

        valid_first = [x for x in first_successes if x is not None]
        mean_first_success = np.mean(valid_first) if valid_first else None

        print(
            f"{label:28s} | "
            f"final-50 success mean = {np.mean(final_success_rates):.3f} | "
            f"final-50 length mean = {np.mean(final_lengths):.1f} | "
            f"mean first success = {mean_first_success}"
        )



def merge_family_runs(*family_runs_dicts):
    """
    Merge multiple {label: histories} dictionaries into one dictionary.
    Useful for cross-family plots.
    """
    merged = {}
    for family_runs in family_runs_dicts:
        merged.update(family_runs)
    return merged


def plot_both_families(sarsa_runs, q_runs):
    """
    Plots all SARSA-family and Q-learning-family algorithms together.
    """
    all_runs = merge_family_runs(sarsa_runs, q_runs)

    plot_with_band(
        all_runs,
        metric="successes",
        transform=cumulative_success,
        smooth_window=None,
        title="All Algorithms: Mean Cumulative Success Rate Across Seeds",
        ylabel="Cumulative Success Rate",
        save_path="all_algorithms_cumulative_success.png",
    )

    plot_with_band(
        all_runs,
        metric="episode_lengths",
        transform=None,
        smooth_window=40,
        title="All Algorithms: Mean Smoothed Episode Length Across Seeds",
        ylabel="Episode Length",
        save_path="all_algorithms_episode_lengths.png",
    )


def plot_eligibility_trace_comparison(sarsa_runs, q_runs):
    """
    Focused plots comparing the eligibility-trace versions:
    SARSA(lambda) vs Q(lambda).
    """
    trace_runs = {}

    sarsa_label = "SarsaLambda(lambda=0.9)"
    q_label = "QLambda(lambda=0.9)"

    if sarsa_label in sarsa_runs:
        trace_runs[sarsa_label] = sarsa_runs[sarsa_label]
    if q_label in q_runs:
        trace_runs[q_label] = q_runs[q_label]

    plot_with_band(
        trace_runs,
        metric="successes",
        transform=cumulative_success,
        smooth_window=None,
        title="Eligibility Traces: SARSA(lambda) vs Q(lambda)",
        ylabel="Cumulative Success Rate",
        save_path="eligibility_traces_cumulative_success.png",
    )

    plot_with_band(
        trace_runs,
        metric="episode_lengths",
        transform=None,
        smooth_window=40,
        title="Eligibility Traces: Smoothed Episode Length",
        ylabel="Episode Length",
        save_path="eligibility_traces_episode_lengths.png",
    )


# -----------------------------
# Experiment runners
# -----------------------------

def run_agent_over_seeds(agent_cls, agent_kwargs, env_kwargs, seeds):
    run_histories = []

    for seed in seeds:
        print(f"  seed={seed}")
        env = GridWorldEnv(seed=seed, **env_kwargs)
        agent = agent_cls(env, seed=seed, **agent_kwargs)
        history = agent.train()
        run_histories.append(history)

    return run_histories


def run_sarsa_family(seeds, env_kwargs):
    family_runs = {}

    configs = [
        (
            "Sarsa",
            agents.Sarsa,
            dict(alpha=0.1, gamma=0.99, epsilon=0.3, num_episodes=600),
        ),
        (
            "NStepSARSA(n=3)",
            agents.NStepSARSA,
            dict(alpha=0.1, gamma=0.99, epsilon=0.3, num_episodes=600, n_step=3),
        ),
        (
            "SarsaLambda(lambda=0.9)",
            agents.SarsaLambdaEligibilityTraces,
            dict(alpha=0.1, gamma=0.99, epsilon=0.3, num_episodes=600, lam=0.9),
        ),
    ]

    for label, agent_cls, agent_kwargs in configs:
        print(f"\nRunning {label}...")
        family_runs[label] = run_agent_over_seeds(
            agent_cls=agent_cls,
            agent_kwargs=agent_kwargs,
            env_kwargs=env_kwargs,
            seeds=seeds,
        )

    # Main plot: cumulative success rate
    plot_with_band(
        family_runs,
        metric="successes",
        transform=cumulative_success,
        smooth_window=None,
        title="SARSA Family: Mean Cumulative Success Rate Across Seeds",
        ylabel="Cumulative Success Rate",
        save_path="sarsa_family_cumulative_success.png",
    )

    # Focused comparison
    plot_with_band(
        family_runs,
        metric="successes",
        transform=cumulative_success,
        smooth_window=None,
        title="Focused SARSA Comparison: n-step vs lambda",
        ylabel="Cumulative Success Rate",
        save_path="sarsa_nstep_vs_lambda_cumulative.png",
        focus_labels=["NStepSARSA(n=3)", "SarsaLambda(lambda=0.9)"],
    )

    # Efficiency plot
    plot_with_band(
        family_runs,
        metric="episode_lengths",
        transform=None,
        smooth_window=40,
        title="SARSA Family: Mean Smoothed Episode Length Across Seeds",
        ylabel="Episode Length",
        save_path="sarsa_family_episode_lengths.png",
    )

    print_family_summary(family_runs)
    return family_runs


def run_q_family(seeds, env_kwargs):
    family_runs = {}

    configs = [
        (
            "QLearning",
            agents.QLearning,
            dict(alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=600),
        ),
        (
            "QLearningBonus(beta=0.1)",
            agents.QLearningBonus,
            dict(alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=600, beta=0.1),
        ),
        (
            "QLambda(lambda=0.9)",
            agents.QLambda,
            dict(alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=600, lam=0.9),
        ),
    ]

    for label, agent_cls, agent_kwargs in configs:
        print(f"\nRunning {label}...")
        family_runs[label] = run_agent_over_seeds(
            agent_cls=agent_cls,
            agent_kwargs=agent_kwargs,
            env_kwargs=env_kwargs,
            seeds=seeds,
        )

    plot_with_band(
        family_runs,
        metric="successes",
        transform=cumulative_success,
        smooth_window=None,
        title="Q-Learning Family: Mean Cumulative Success Rate Across Seeds",
        ylabel="Cumulative Success Rate",
        save_path="q_family_cumulative_success.png",
    )

    plot_with_band(
        family_runs,
        metric="episode_lengths",
        transform=None,
        smooth_window=40,
        title="Q-Learning Family: Mean Smoothed Episode Length Across Seeds",
        ylabel="Episode Length",
        save_path="q_family_episode_lengths.png",
    )

    print_family_summary(family_runs)
    return family_runs


# -----------------------------
# Main
# -----------------------------

def main():
    # Use multiple seeds to reduce luck/noise
    seeds = [0, 1, 2, 3, 4]

    # Harder environment so n-step SARSA vs SARSA(lambda) is more visible
    env_kwargs = dict(
        size=15,
        reward_probability=0.01,
    )

    print("Environment settings:")
    print(env_kwargs)
    print("Seeds:", seeds)

    print("\n=== Running SARSA-family comparison ===")
    sarsa_runs = run_sarsa_family(seeds=seeds, env_kwargs=env_kwargs)

    print("\n=== Running Q-family comparison ===")
    q_runs = run_q_family(seeds=seeds, env_kwargs=env_kwargs)

    print("\n=== Running combined cross-family plots ===")
    plot_both_families(sarsa_runs, q_runs)

    print("\n=== Running eligibility-trace focused plots ===")
    plot_eligibility_trace_comparison(sarsa_runs, q_runs)

    print("\nDone.")
    print("Saved:")
    print(" - sarsa_family_cumulative_success.png")
    print(" - sarsa_nstep_vs_lambda_cumulative.png")
    print(" - sarsa_family_episode_lengths.png")
    print(" - q_family_cumulative_success.png")
    print(" - q_family_episode_lengths.png")
    print(" - all_algorithms_cumulative_success.png")
    print(" - all_algorithms_episode_lengths.png")
    print(" - eligibility_traces_cumulative_success.png")
    print(" - eligibility_traces_episode_lengths.png")


if __name__ == "__main__":
    main()