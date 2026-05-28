from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from snake_game import SnakeGame
from utils import plot_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "snake_dqn.pth"


def train(num_games: int, model_path: Path, plot: bool = True) -> None:
    scores = []
    mean_scores = []
    total_score = 0
    record = -1

    agent = Agent()
    game = SnakeGame(render=False)

    try:
        while agent.n_games < num_games:
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, explore=True)

            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)

            # Short memory trains on the single transition that just happened.
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1

                # Long memory trains on a random batch from replay memory.
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.save_model(model_path)

                total_score += score
                mean_score = total_score / agent.n_games
                scores.append(score)
                mean_scores.append(mean_score)

                print(
                    f"Game {agent.n_games} "
                    f"Score {score} "
                    f"Record {record} "
                    f"Epsilon {agent.epsilon:.3f}"
                )

                if plot:
                    plot_scores(scores, mean_scores)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    finally:
        game.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake.")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to train.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the best model.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the live matplotlib plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(num_games=args.games, model_path=args.model_path, plot=not args.no_plot)
