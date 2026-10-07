# Lunar Lander DQN Agent

Deep Q-Network agent trained on OpenAI Gymnasium's LunarLander-v2 environment.

## Features
- Type hints throughout codebase
- Comprehensive docstrings (Google style)
- Structured logging (not print statements)
- Replay buffer for experience sampling
- Target network for stable Q-learning
- Soft target network updates (TAU=0.005)

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

- **QNetwork**: Simple feedforward network (state → Q-values)
- **ReplayBuffer**: Stores and samples past experiences
- **DQNAgent**: Orchestrates learning loop with epsilon-greedy exploration

## Key Hyperparameters

- GAMMA = 0.99 (discount factor)
- EPS_START = 1.0, EPS_END = 0.01, EPS_DECAY = 0.995
- BATCH_SIZE = 64
- TAU = 0.005 (soft update rate)

## Testing

Run the full test suite with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Status: 100% coverage (13 tests passing)**

Test suite covers:
- QNetwork forward pass and initialisation
- ReplayBuffer remember, sample, and capacity limits
- DQNAgent initialisation, exploration/exploitation, replay training, epsilon decay, save/load

Tests run automatically via GitHub Actions on every push ('.github/workflows/tests.yml').

## Code Quality

- Type hints: ✅
- Docstrings: ✅
- Logging: ✅
- Refactored helpers: ✅
- Tests: ✅ (100% coverage)