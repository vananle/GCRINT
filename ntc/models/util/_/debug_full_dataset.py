
# Debug full dataset
import torch

X = dataloader_full.dataset.X_scaled
with torch.no_grad():
    Xhat = []
    for batch in dataloader_full:
        Xhat.append(batch['x'])
    Xhat_scaled = torch.cat(Xhat).reshape(tuple(dataloader_full.dataset.X.shape))
    Xhat = dataloader_full.dataset.scaler.invert_transform(Xhat_scaled)

print(torch.sum(X - Xhat_scaled))
