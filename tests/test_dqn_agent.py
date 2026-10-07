import pytest
import numpy as np
import torch
from src.dqn_agent import QNetwork, ReplayBuffer, DQNAgent


class TestQNetwork:
    @pytest.fixture
    def network(self):
        return QNetwork(state_dim=8, action_dim=4)

    def test_forward_pass(self, network):
        """Test QNetwork forward pass."""
        state = torch.randn(1, 8)
        q_values = network(state)
        assert q_values.shape == (1, 4)

    def test_initialization(self, network):
        """Test QNetwork initialization."""
        assert network.layer1.in_features == 8
        assert network.layer2.in_features == 64
        assert network.layer3.out_features == 4


class TestReplayBuffer:
    @pytest.fixture
    def buffer(self):
        return ReplayBuffer(capacity=100)

    def test_remember(self, buffer):
        """Test storing experience in buffer."""
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action = 0
        reward = 1.0
        next_state = np.array([2.0, 3.0, 4.0, 5.0])
        done = False

        buffer.remember(state, action, reward, next_state, done)
        assert len(buffer) == 1

    def test_sample(self, buffer):
        """Test sampling from buffer."""
        for i in range(10):
            state = np.random.randn(8).astype(np.float32)
            buffer.remember(state, i % 4, float(i), state + 0.1, False)

        batch = buffer.sample(batch_size=5)
        states, actions, rewards, next_states, dones = batch
        assert states.shape == (5, 8)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 8)
        assert dones.shape == (5,)

    def test_buffer_capacity(self):
        """Test buffer respects max capacity."""
        buffer = ReplayBuffer(capacity=5)
        for i in range(10):
            buffer.remember(np.array([1.0]), 0, 1.0, np.array([2.0]), False)
        assert len(buffer) == 5


class TestDQNAgent:
    @pytest.fixture
    def agent(self):
        return DQNAgent(state_dim=8, action_dim=4, device="cpu")

    def test_initialization(self, agent):
        """Test DQNAgent initialization."""
        assert agent.state_dim == 8
        assert agent.action_dim == 4
        assert agent.epsilon == agent.EPS_START
        assert agent.episode == 0

    def test_act_exploration(self, agent):
        """Test act() explores with high epsilon."""
        agent.epsilon = 1.0
        state = np.random.randn(8).astype(np.float32)
        action = agent.act(state)
        assert 0 <= action < 4

    def test_act_exploitation(self, agent):
        """Test act() exploits with low epsilon."""
        agent.epsilon = 0.0
        state = np.random.randn(8).astype(np.float32)
        action = agent.act(state)
        assert 0 <= action < 4

    def test_remember(self, agent):
        """Test storing experience."""
        state = np.random.randn(8).astype(np.float32)
        agent.remember(state, 0, 1.0, state + 0.1, False)
        assert len(agent.memory) == 1

    def test_replay_insufficient_buffer(self, agent):
        """Test replay() returns None with insufficient buffer."""
        result = agent.replay()
        assert result is None

    def test_replay_sufficient_buffer(self, agent):
        """Test replay() trains with sufficient buffer."""
        for i in range(1100):
            state = np.random.randn(8).astype(np.float32)
            agent.remember(state, i % 4, float(i % 10), state + 0.1, False)

        agent.episode = 100
        loss = agent.replay()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss > 0

    def test_decay_epsilon(self, agent):
        """Test epsilon decay."""
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial_epsilon
        assert agent.episode == 1

    def test_save_load(self, agent, tmp_path):
        """Test saving and loading checkpoint."""
        filepath = tmp_path / "checkpoint.pt"
        agent.save(str(filepath))
        assert filepath.exists()

        new_agent = DQNAgent(state_dim=8, action_dim=4, device="cpu")
        new_agent.load(str(filepath))
        assert new_agent.epsilon == agent.epsilon
