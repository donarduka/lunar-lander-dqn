import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

""" Q-network - simple feedforward network that takes the state as input and outputs
    Q-values for each action"""
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
""" Replay Buffer - stores past experiences so the agent can learn from them later
    sampling randomly breaks the correlation between consecutive steps"""
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype = np.float32),
            np.array(actions, dtype = np.int64),
            np.array(rewards, dtype = np.float32),
            np.array(next_states, dtype = np.float32),
            np.array(dones, dtype = np.float32),
        )
    
    def __len__(self):
        return len(self.memory)
    
""" DQN Agent - brings the Q-network, target network replay buffer and epsilon-greedy
    exploration together"""
class DQNAgent:
    #hyperparameters
    BATCH_SIZE = 64 # number of transitions samples from replay buffer
    GAMMA = 0.99 # discount factor - how valued future rewards are
    EPS_START = 1.0 # starting exploration rate
    EPS_END = 0.01 # min exploration rate
    EPS_DECAY = 0.995 # how quickly epsilon decays each episode
    LR = 5e-4 # learning rate for Adam optimiser
    TAU = 0.005 # soft update rate for target network
    BUFFER_SIZE = 50000 # max transitions stored in replay buffer
    MIN_BUFFER = 1000 # min buffer size before training starts

    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu') -> None:
        """Initialise DQN Agent with policy and target networks.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.epsilon = self.EPS_START
        self.episode = 0

        #online network i.e. what is being trained
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        #target network - used to compute stable TD targets
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayBuffer(self.BUFFER_SIZE)
        self.losses = []

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer.
        
        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state
            done: episode termination flag
        """
        self.memory.remember(state, action, reward, next_state, done)

    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy.
        
        Args:
            state: current state
            
        Returns:
            selected action index
        """
        #eps-greedy - explore randomly or exploit best known action
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(1).item())

    def _convert_to_tensors(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> tuple:
        """ Convert numpy arrays to PyTorch tensors on device
        
        Args:
            states: State array (batch_size, state_dim)
            actions: Action array (batch_size,)
            rewards: Reward array (batch_size,)
            next_states: Next state array (batch_size, state_dim)
            dones: Done flags (batch_size,)
        
        Returns:
            Tuple of tensors on device
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def _comput_q_values(self, states_t: torch.Tensor, actions_t: torch.Tensor, next_states_t: torch.Tensor, dones_t: torch.Tensor, rewards_t: torch.Tensor) -> tuple:
        """Compute current and target Q-values using Bellman equation.

        Args:
            states_t: State tensors (batch_size, state_dim)
            actions_t: Action tensors (batch_size, 1)
            next_states_t: Next state tensors (batch_size, state_dim)
            dones_t: Done flags (batch_size, 1)
            rewards_t: Reward tensors (batch_size, 1)

        Returns:
            Tuple of (current_q, target_q) tensors
        """
        # Current Q-values for actions taken
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q-values using Bellman equation
        # Target network provides stable Q estimates for next states
        with torch.no_grad():
            next_q = self.target_net(next_states_t).mac(1, keepdim=True).values
            target_q = rewards_t + self.GAMMA * next_q * (1 - dones_t)

        return current_q, target_q

    def _update_network(self, current_q: torch.Tensor, target_q: torch.Tensor) -> float:
        """Compute loss, backprop and update network weights.

        Args:
            current_q: Current Q-value predictions (batch_size, 1)
            target_q: Target Q-values from Bellman (batch_size, 1)

        Returns:
            Loss value as float
        """
        # Huber loss - less sensitive to outlier than MSE
        # Works like MSE for small errors, MAE for large errors
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q, target_q)

        # Backward pass
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()

        # Soft update target network (θ - τθ + (1-τ)θ')
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.TAU * policy_param.data + (1 - self.TAU) * target_param.data)

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def replay(self):
        #dont train until there are enough experiences
        if len(self.memory) < self.MIN_BUFFER:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)
        states_t, actions_t, rewards_t, next_states_t, dones_t = self._convert_to_tensors(states, actions, rewards, next_states, dones)
        current_q, target_q = self._comput_q_values(states_t, actions_t, next_states_t, dones_t, rewards_t)
        loss_val = self._update_network(current_q, target_q)
        return loss_val
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate and increment episode counter."""
        #decay epsilon at the end of each epsiode
        self.epsilon = max(self.EPS_END, self.epsilon * self.EPS_DECAY)
        self.episode += 1

    def save(self, path: str) -> None:
        """Save model checkpoint to disk.
        
        Args:
            path: File path to save checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode,
            'losses': self.losses
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint from disk.
        
        Args:
            path: File path to load checkpoint from
        """
        data = torch.load(path, mpa_location=self.device)
        self.policy_net.load_state_dict(data['policy_net'])
        self.target_net.load_state_dict(data['target_net'])
        self.optimiser.load_state_dict(data['optimiser'])
        self.epsilon = data['epsilon']
        self.episode = data['episode']
        self.losses = data['losses']
        print(f"Model loaded from {path}")