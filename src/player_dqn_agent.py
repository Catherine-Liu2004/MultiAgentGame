import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    ACTIONS,
    PLAYER_BATCH_SIZE,
    PLAYER_EPSILON,
    PLAYER_EPSILON_DECAY,
    PLAYER_EPSILON_MIN,
    PLAYER_GAMMA,
    PLAYER_LR,
    PLAYER_MEMORY_SIZE,
    PLAYER_TARGET_UPDATE,
)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class PlayerDQNAgent:
    def __init__(self, state_dim, action_dim=None, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim if action_dim is not None else len(ACTIONS)

        self.gamma = PLAYER_GAMMA
        self.epsilon = PLAYER_EPSILON
        self.epsilon_min = PLAYER_EPSILON_MIN
        self.epsilon_decay = PLAYER_EPSILON_DECAY
        self.batch_size = PLAYER_BATCH_SIZE
        self.target_update = PLAYER_TARGET_UPDATE

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=PLAYER_LR)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=PLAYER_MEMORY_SIZE)
        self.learn_step = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([b[0] for b in batch], dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(np.array([b[1] for b in batch], dtype=np.int64)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch], dtype=np.float32)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch], dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch], dtype=np.float32)).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min