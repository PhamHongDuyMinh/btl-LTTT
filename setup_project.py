import os


project_structure = {
    "utils": {},
    "models": {},
    "data": {},
}

files_content = {
    "requirements.txt": """torch
numpy
matplotlib
seaborn
""",

    "utils/mine_loss.py": """import torch
import torch.nn as nn

def mine_lower_bound(t, t_marginal):
    return torch.mean(t) - torch.log(torch.mean(torch.exp(t_marginal)))

def mutual_information_loss(network, x, y):
    t_joint = network(x, y)
    y_shuffle = y[torch.randperm(y.shape[0])]
    t_marginal = network(x, y_shuffle)
    mi_score = mine_lower_bound(t_joint, t_marginal)
    return -mi_score, mi_score
""",

    "models/network.py": """import torch
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
""",

    "data/gaussian.py": """import numpy as np

def sample_correlated_gaussian(batch_size, rho=0.9, dim=1):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, size=batch_size * dim)
    x = data[:, 0].reshape(batch_size, dim)
    y = data[:, 1].reshape(batch_size, dim)
    return x, y

def true_mi_gaussian(rho, dim):
    return -0.5 * np.log(1 - rho**2) * dim
""",

    "main.py": """import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from models.network import MineNetwork
from utils.mine_loss import mutual_information_loss
from data.gaussian import sample_correlated_gaussian, true_mi_gaussian

BATCH_SIZE = 512
LR = 1e-3
ITERATIONS = 5000
RHO = 0.9
DIM = 1

model = MineNetwork(input_dim=DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)

mi_history = []
true_mi = true_mi_gaussian(RHO, DIM)

print(f"Start Training MINE... True MI: {true_mi:.4f}")

for i in range(ITERATIONS):
    x_np, y_np = sample_correlated_gaussian(BATCH_SIZE, rho=RHO, dim=DIM)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    optimizer.zero_grad()
    loss, mi_est = mutual_information_loss(model, x, y)
    loss.backward()
    optimizer.step()
    
    mi_history.append(mi_est.item())
    
    if i % 500 == 0:
        print(f"Step {i}, Estimated MI: {mi_est.item():.4f}")

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure(figsize=(10, 5))
plt.plot(mi_history, label='MINE Estimated', alpha=0.3)
plt.plot(moving_average(mi_history), label='MINE Smoothed', linewidth=2)
plt.axhline(y=true_mi, color='r', linestyle='--', label='True MI')
plt.title(f'MINE Estimation (Rho={RHO})')
plt.legend()
plt.savefig('result.png')
print("Training done. Result saved to result.png")
""",

    "README.md": """# MINE Estimation - HUST Project
Dự án tìm hiểu và cài đặt Mutual Information Neural Estimation (ICML 2018).

## Cài đặt
`pip install -r requirements.txt`

## Chạy chương trình
`python main.py`
"""
}

def create_project():
    # Tạo thư mục
    for folder in project_structure:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
    
    # Tạo file
    for filepath, content in files_content.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {filepath}")

if __name__ == "__main__":
    create_project()