import torch
import torch.nn as nn

def mine_lower_bound(t, t_marginal):
    return torch.mean(t) - torch.log(torch.mean(torch.exp(t_marginal)))

def mutual_information_loss(network, x, y):
    t_joint = network(x, y)
    y_shuffle = y[torch.randperm(y.shape[0])]
    t_marginal = network(x, y_shuffle)
    mi_score = mine_lower_bound(t_joint, t_marginal)
    return -mi_score, mi_score
