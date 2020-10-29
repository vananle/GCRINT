# Debug: missing dataset
import torch

for batch in dataloader:
    X, I, J, K = batch['x'], batch['i'], batch['j'], batch['k']
    for _ in range(len(X)):
        x = X[_]
        x_correct = dataloader.dataset.X_scaled[I[_], J[_], K[_]]
        print(float(x), float(x_correct))
    break
