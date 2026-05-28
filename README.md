# Snake AI

A small Deep Q-Network project that trains an agent to play Snake.

The environment is written from scratch in Python. PyTorch powers the model, and
Pygame is only used when you watch the trained agent play.

## Project Structure

```text
snake-ai/
  models/
    snake_dqn_simple.pth
    snake_dqn_grid.pth
  src/
    agent.py
    model.py
    play.py
    snake_game.py
    train.py
    utils.py
  requirements.txt
```

## Requirements

- Python 3.10 or newer
- pip

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train The Agent

The training script `src/train.py` supports multiple state modes, device routing, vectorized environment batching, and custom hyperparameters.

### State Modes
- `--state-mode simple` (default): Hand-crafted 11-value local feature vector. Compact and extremely fast to train. Saves to `models/snake_dqn_simple.pth`.
- `--state-mode grid`: Full-board vision vector (size 400 for a 20x20 board) representing every cell normalized (0.0: empty, 0.33: body, 0.66: head, 1.0: food). Saves to `models/snake_dqn_grid.pth`. Resuming from a checkpoint automatically checks compatibility.

### Parallel Vectorized Environments
- `--num-envs` (default 1): Collects experiences faster by running multiple environments concurrently. Epsilon exploration is processed independently, while model inference and short-memory training are fully batched into single PyTorch tensor operations. Note that this is vectorized training batching, not OS-level multiprocessing.

### Commands

**Train locally in simple mode (default 1000 games, single environment):**
```bash
python src/train.py
```

**Train on Google Colab (high-speed GPU batch training in grid mode with 16 parallel environments):**
```bash
python src/train.py --state-mode grid --games 20000 --num-envs 16 --device cuda --no-plot
```

**Train with custom batch sizes, environments, and checkpoints:**
```bash
python src/train.py --state-mode grid --games 5000 --num-envs 8 --batch-size 1000 --checkpoint-every 1000 --no-plot
```

### Save Locations and Checkpoints
- The best model is saved to `models/snake_dqn_simple.pth` or `models/snake_dqn_grid.pth` depending on the `--state-mode`.
- Periodic checkpoints are saved every N games (configured by `--checkpoint-every`, default 500) under:
  ```text
  models/snake_dqn_simple_checkpoint_500.pth
  models/snake_dqn_grid_checkpoint_1000.pth
  ```
- Resuming automatically detects shape/state_mode mismatches in checkpoints and warns before starting fresh to prevent crashes.

## Watch The AI Play

Watch the trained Agent play in Pygame.

**Watch local play in simple mode:**
```bash
python src/play.py --state-mode simple
```

**Watch local play in grid mode (CPU laptop-friendly):**
```bash
python src/play.py --state-mode grid --model-path models/snake_dqn_grid.pth --device cpu
```

**Change playback speed:**
```bash
python src/play.py --state-mode grid --speed 40
```

## How It Works

The project implements a deep reinforcement learning agent using Q-learning.

### State Modes
1. **Simple Mode**:
   The agent observes an 11-value state vector including:
   - Nearby collision danger in three directions (straight, right, left)
   - Current direction vectors (moving right, left, up, down)
   - Relative food position (food left, right, up, down)
2. **Grid Mode**:
   The agent observes a full-board vision state (e.g. 400 inputs for a 20x20 board) representing every cell:
   - `0.0` = empty space
   - `0.33` = snake body segment
   - `0.66` = snake head
   - `1.0` = food

### Actions
At each step, the agent chooses one of three relative actions:
- `0` = go straight
- `1` = turn right
- `2` = turn left

### Rewards
- `+10.0` for eating food.
- `-10.0` for dying (crashing into a wall, tail, or starvation timeout).
- `-0.01` per move to encourage fast paths.
- **Simple Mode Only**: Small auxiliary rewards are added to guide the snake:
  - `+0.05` for moving closer to food
  - `-0.05` for moving farther from food

### Starvation Timeout
Episodes automatically terminate with a `-10` penalty if the snake goes more than `100 * len(snake)` steps without eating. This prevents loops where the snake circles endlessly.

### Network Architecture
The agent uses a Deep Q-Network (DQN) with a multi-layer perceptron (MLP) architecture:
`input_size -> 256 -> 256 -> 128 -> 3` (ReLU activations between hidden layers).
- For simple mode, `input_size = 11`.
- For grid mode, `input_size = width * height` (e.g., 400).

Training leverages high-performance GPUs (via CUDA) to compute gradient descents on large batches of experiences. Playback runs on lightweight CPU structures via safe model tensor mapping.
