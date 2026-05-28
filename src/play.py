from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from agent import Agent
from snake_game import SnakeGame


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def play(model_path: Path, state_mode: str, device: str, speed: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train first with: "
            f"python src/train.py --state-mode {state_mode}"
        )

    # Initialize the Agent on the specified device (CPU by default to be laptop-friendly)
    agent = Agent(state_mode=state_mode, device=device)
    
    # Load model (uses map_location=device under the hood inside agent.load_model)
    agent.load_model(model_path)
    
    # Explicitly guarantee evaluation mode
    agent.model.eval()
    
    game = SnakeGame(render=True, speed=speed, state_mode=state_mode)

    try:
        while True:
            state = agent.get_state(game)
            
            # Predict action under torch.no_grad() to save memory and avoid tracking gradients
            with torch.no_grad():
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
        "--state-mode",
        choices=["simple", "grid"],
        default="simple",
        help="State representation mode: 'simple' (11-value vector) or 'grid' (400-value board vision).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Device to run inference on: 'cpu' (default, highly recommended for laptops), 'auto', or 'cuda'.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the saved model file. Defaults based on state-mode if omitted.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=20,
        help="Pygame playback refresh speed (frames per second).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Dynamically resolve default model path based on state-mode if not explicitly provided
    if args.model_path is None:
        if args.state_mode == "simple":
            resolved_model_path = PROJECT_ROOT / "models" / "snake_dqn_simple.pth"
        else:
            resolved_model_path = PROJECT_ROOT / "models" / "snake_dqn_grid.pth"
    else:
        resolved_model_path = args.model_path

    play(
        model_path=resolved_model_path,
        state_mode=args.state_mode,
        device=args.device,
        speed=args.speed,
    )
