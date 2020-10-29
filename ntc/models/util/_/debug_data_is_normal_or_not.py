
    # Check if data is normally distributed or not
    import torch
    import matplotlib.pyplot as plt

    X = dataloader.dataset.X_scaled.reshape(-1, 144)
    plt.plot(torch.mean(X, dim=0).data.numpy(), label='mean')
    plt.plot(torch.std(X, dim=0).data.numpy(), label='std')
    plt.legend()
    plt.show()
