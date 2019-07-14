import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from Model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, bufferSize, batchSize, gamma, tau, lr, updateEvery, state_size, action_size, seed):
        # hyper parameters
        self.BufferSize = bufferSize
        self.BatchSize = batchSize
        self.Gamma = gamma
        self.Tau = tau
        self.LR = lr
        self.UpdateEvery = updateEvery
        self.State_size = state_size
        self.Action_size = action_size
        self.Seed = random.seed(seed)

        # Q network
        self.QNetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.QNetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.Optimizer = optim.Adam(self.QNetwork_local.parameters(), lr=self.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BufferSize, self.BatchSize, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.QNetwork_local.eval()
        with torch.no_grad():
            action_values = self.QNetwork_local(state)
        self.QNetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.Action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UpdateEvery
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BatchSize:
                experiences = self.memory.sample()
                self.learn(experiences, self.Gamma)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.QNetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.QNetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.Optimizer.zero_grad()
        loss.backward()
        self.Optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.QNetwork_local, self.QNetwork_target, self.Tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.Action_size = action_size
        self.Memory = deque(maxlen=buffer_size)
        self.Batch_size = batch_size
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.Seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.Experience(state, action, reward, next_state, done)
        self.Memory.append(e)

    def sample(self):
        experiences = random.sample(self.Memory, k=self.Batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.Memory)
