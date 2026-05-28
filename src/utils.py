from __future__ import annotations

import matplotlib.pyplot as plt


def plot_scores(scores: list[int], mean_scores: list[float]) -> None:
    """Update a live matplotlib plot during training."""

    plt.ion()
    plt.clf()
    plt.title("Snake DQN Training")
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean score")
    plt.legend(loc="upper left")

    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")

    plt.pause(0.001)
