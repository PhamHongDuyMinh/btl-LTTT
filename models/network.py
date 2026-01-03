import torch
import torch.nn as nn

class MineNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(MineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
