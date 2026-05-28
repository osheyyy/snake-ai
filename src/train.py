from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from snake_game import SnakeGame
from utils import plot_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLLING_AVERAGE_INTERVAL = 100


def checkpoint_path_for(model_path: Path, game_number: int) -> Path:
    return model_path.with_name(f"{model_path.stem}_checkpoint_{game_number}{model_path.suffix}")


def resume_if_available(agent: Agent, model_path: Path) -> bool:
    if not model_path.exists():
        return False

    try:
        agent.load_model(model_path)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"WARNING: Could not resume from {model_path}. The checkpoint is incompatible or unreadable.")
        print(f"Error detail: {e}")
        print("Starting with a fresh model.")
        return False

    agent.model.train()
    print(f"Resumed model from {model_path}")
    return True


def train(
    num_games: int,
    model_path: Path,
    state_mode: str,
    device: str,
    batch_size: int,
    checkpoint_every: int,
    num_envs: int = 1,
    plot: bool = True,
) -> None:
    scores = []
    mean_scores = []
    total_score = 0
    record = -1

    # Instantiate Agent with specified dynamic settings
    agent = Agent(
        state_mode=state_mode,
        device=device,
        batch_size=batch_size,
    )
    
    resume_if_available(agent, model_path)
    
    if num_envs == 1:
        # Keep original single environment training loop behavior
        game = SnakeGame(render=False, state_mode=state_mode)
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
                        print(f"New Record! Saved best model to {model_path}")

                    if agent.n_games % checkpoint_every == 0:
                        checkpoint_path = checkpoint_path_for(model_path, agent.n_games)
                        agent.save_model(checkpoint_path)
                        print(f"Saved periodic checkpoint to {checkpoint_path}")

                    total_score += score
                    mean_score = total_score / agent.n_games
                    scores.append(score)
                    mean_scores.append(mean_score)
                    rolling_average = sum(scores[-ROLLING_AVERAGE_INTERVAL:]) / min(
                        len(scores),
                        ROLLING_AVERAGE_INTERVAL,
                    )

                    print(
                        f"Game {agent.n_games} | "
                        f"Score {score} | "
                        f"Record {record} | "
                        f"Epsilon {agent.epsilon:.3f}"
                    )

                    if agent.n_games % ROLLING_AVERAGE_INTERVAL == 0:
                        print(
                            f"--- Last {ROLLING_AVERAGE_INTERVAL} games "
                            f"average score: {rolling_average:.2f} ---"
                        )

                    if plot:
                        plot_scores(scores, mean_scores)
        except KeyboardInterrupt:
            print("\nTraining stopped by user.")
        finally:
            game.close()
    else:
        # Vectorized/parallelized environment training style batching
        envs = [SnakeGame(render=False, state_mode=state_mode) for _ in range(num_envs)]
        try:
            while agent.n_games < num_games:
                # 1. Batch state extraction from all parallel environment instances
                states_old = [envs[i].get_state() for i in range(num_envs)]
                
                # 2. Batch action prediction in one single neural net tensor pass
                actions = agent.get_actions(states_old, explore=True)
                
                rewards = []
                dones = []
                states_new = []
                
                # 3. Step all environments concurrently
                for i in range(num_envs):
                    reward, done, score = envs[i].play_step(actions[i])
                    state_new = envs[i].get_state()
                    
                    rewards.append(reward)
                    dones.append(done)
                    states_new.append(state_new)
                
                # 4. Train short memory as a single combined batch of N transitions
                agent.train_short_memory(states_old, actions, rewards, states_new, dones)
                
                # 5. Append all transitions to the experience replay memory
                for i in range(num_envs):
                    agent.remember(states_old[i], actions[i], rewards[i], states_new[i], dones[i])
                    
                    # 6. Process game completions sequentially as they occur
                    if dones[i]:
                        score = envs[i].score
                        envs[i].reset()
                        agent.n_games += 1
                        
                        # Trigger long memory replay training
                        agent.train_long_memory()
                        
                        # Update performance records and save best models
                        if score > record:
                            record = score
                            agent.save_model(model_path)
                            print(f"New Record! Saved best model to {model_path}")
                            
                        if agent.n_games % checkpoint_every == 0:
                            checkpoint_path = checkpoint_path_for(model_path, agent.n_games)
                            agent.save_model(checkpoint_path)
                            print(f"Saved periodic checkpoint to {checkpoint_path}")
                            
                        total_score += score
                        mean_score = total_score / agent.n_games
                        scores.append(score)
                        mean_scores.append(mean_score)
                        rolling_average = sum(scores[-ROLLING_AVERAGE_INTERVAL:]) / min(
                            len(scores),
                            ROLLING_AVERAGE_INTERVAL,
                        )
                        
                        print(
                            f"Game {agent.n_games} | "
                            f"Score {score} | "
                            f"Record {record} | "
                            f"Epsilon {agent.epsilon:.3f}"
                        )
                        
                        if agent.n_games % ROLLING_AVERAGE_INTERVAL == 0:
                            print(
                                f"--- Last {ROLLING_AVERAGE_INTERVAL} games "
                                f"average score: {rolling_average:.2f} ---"
                            )
                            
                        if plot:
                            plot_scores(scores, mean_scores)
                            
                        # Terminate early if target game count is reached within loop
                        if agent.n_games >= num_games:
                            break
        except KeyboardInterrupt:
            print("\nTraining stopped by user.")
        finally:
            for env in envs:
                env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake.")
    parser.add_argument(
        "--state-mode",
        choices=["simple", "grid"],
        default="simple",
        help="State representation mode: 'simple' (11-value vector) or 'grid' (400-value board vision).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to train on: 'auto' (CUDA if available), 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to train.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to step and collect training experiences concurrently.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for DQN training updates.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Checkpoint save interval (in games).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the live matplotlib visualization plot.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Custom path to save the best model. Defaults based on state-mode.",
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

    train(
        num_games=args.games,
        model_path=resolved_model_path,
        state_mode=args.state_mode,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        num_envs=args.num_envs,
        plot=not args.no_plot,
    )
