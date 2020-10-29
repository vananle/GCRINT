
#    # Debug X vs invert_transform(x)
#    X = dataloader_full.dataset.X
#    # print('original data', X[:100, 1])
#    X_scaled = dataloader_full.dataset.X_scaled
#    Xhat = dataloader_full.dataset.scaler.invert_transform(X_scaled)
#    X = X.reshape(Xhat.shape)
#    X_scaled = X_scaled.reshape(Xhat.shape)
#    # print('after invert_transform', Xhat[:100, 1])
#    # print('MAE error', util.mae_normal(X, Xhat))
#
#    # plot X
#    import matplotlib.pyplot as plt
#    plt.subplot(131)
#    plt.plot(X[:])
#    plt.title('X')
#    plt.subplot(132)
#    plt.plot(Xhat[:])
#    plt.title('Xhat')
#    plt.subplot(133)
#    plt.plot(X_scaled[:])
#    plt.title('X_scaled')
#    plt.show()


    # Debug X vs invert_transform(x)
    import matplotlib.pyplot as plt
    X = dataloader_full.dataset.X
    X = X.reshape(-1, 144)
    plt.plot(X[:, 1], label='original data')
    X_scaled = dataloader_full.dataset.X_scaled
    Xhat = dataloader_full.dataset.scaler.invert_transform(X_scaled)
    plt.plot(Xhat[:, 1], label='transform then inverted data')
    plt.legend()
    plt.title('MAE error = {}'.format(util.mae_normal(X, Xhat)))
    plt.show()
