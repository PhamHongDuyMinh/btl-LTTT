import numpy as np

def sample_correlated_gaussian(batch_size, rho=0.9, dim=1):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, size=batch_size * dim)
    x = data[:, 0].reshape(batch_size, dim)
    y = data[:, 1].reshape(batch_size, dim)
    return x, y

def true_mi_gaussian(rho, dim):
    return -0.5 * np.log(1 - rho**2) * dim
