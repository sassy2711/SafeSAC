import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-6

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))

        mu = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()

        # Reparameterization trick
        epsilon = torch.randn_like(std)
        u = mu + std * epsilon

        # Apply tanh squashing
        a = torch.tanh(u)

        # Gaussian log prob (before tanh)
        log_prob = -0.5 * (
            ((u - mu) / (std + EPS))**2 + 2 * log_std + np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Tanh correction
        log_prob -= torch.sum(
            torch.log(1 - a.pow(2) + EPS), dim=-1, keepdim=True
        )

        return a, log_prob