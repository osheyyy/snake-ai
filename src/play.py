from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent import Agent
from snake_game import SnakeGame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "snake_dqn.pth"


def play(model_path: Path, speed: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train first with: python src/train.py"
        )

    agent = Agent()
    agent.load_model(model_path)
    game = SnakeGame(render=True, speed=speed)

    try:
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state, explore=False)
            _, done, score = game.play_step(action)

            if done:
                print(f"Game over. Score: {score}")
                time.sleep(0.5)
                game.reset()
    finally:
        game.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the trained Snake AI.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model.",
    )
    parser.add_argument("--speed", type=int, default=20, help="Pygame playback speed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play(model_path=args.model_path, speed=args.speed)
