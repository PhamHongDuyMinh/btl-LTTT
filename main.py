import torch
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
