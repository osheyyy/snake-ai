# Snake AI

A small Deep Q-Network project that trains an agent to play Snake.

The environment is written from scratch in Python. PyTorch powers the model, and
Pygame is only used when you watch the trained agent play.

## Project Structure

```text
snake-ai/
  models/
    snake_dqn.pth
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

Train for the default 1000 games:

```bash
python src/train.py
```

Train for a custom number of games:

```bash
python src/train.py --games 2000
```

Run training without the live matplotlib chart:

```bash
python src/train.py --no-plot
```

If `models/snake_dqn.pth` already exists, training automatically tries to load
it before starting. The best model is saved back to the same path when the record
score improves.

The best model is saved to:

```text
models/snake_dqn.pth
```

Training also saves periodic checkpoints every 500 games:

```text
models/snake_dqn_checkpoint_500.pth
models/snake_dqn_checkpoint_1000.pth
```

## Watch The AI Play

After training, run:

```bash
python src/play.py
```

Change playback speed:

```bash
python src/play.py --speed 40
```

## How It Works

The agent observes an 11-value state, including nearby collision danger, current
direction, and food position. It chooses one of three relative actions:

```text
0 = go straight
1 = turn right
2 = turn left
```

Rewards are simple:

```text
+10    eat food
-10    die or get stuck too long
-0.01  every move
+0.05  move closer to food
-0.05  move farther from food
```

Episodes also end when the snake spends more than `100 * len(snake)` frames
without making progress. During training, the agent uses epsilon-greedy
exploration and replay memory. The neural network has two hidden layers
(`11 -> 128 -> 128 -> 3`) and learns Q-values for each action.
