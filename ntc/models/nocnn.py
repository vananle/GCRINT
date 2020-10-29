import torch
from . import util
from tqdm import tqdm

class NoCNN(torch.nn.Module):

    def __init__(self, args):
        super(NoCNN, self).__init__()
        # save args
        self.args = args

        # parameter
        rank  = 16
        alpha = 1e-2
        beta  = 1e-2
        gamma = 1e-2

        # save parameter
        self.rank = rank

        # initialize component
        self.add_cp_layer()

        # linear layer
        self.linear_layer = torch.nn.Linear(rank, 1, bias=True)

        # create optimizer
        self.optimizer = util.get_optimizer(self, args.learning_rate)

    def add_cp_layer(self):
        A = torch.randn(self.args.time_length, self.rank)
        B = torch.randn(self.args.dim, self.rank)
        C = torch.randn(self.args.dim, self.rank)
        A = torch.nn.Parameter(A)
        B = torch.nn.Parameter(B)
        C = torch.nn.Parameter(C)
        self.register_parameter('A', A)
        self.register_parameter('B', B)
        self.register_parameter('C', C)
        self.A = A
        self.B = B
        self.C = C

    def forward(self, batch, verbose=False):
        # extract parameter
        args = self.args
        batch_size = len(batch['i'])
        a = self.A[batch['i']]
        b = self.B[batch['j']]
        c = self.C[batch['k']]

        # build interation map
        a = a.unsqueeze(2)
        b = b.unsqueeze(1)
        c = c.unsqueeze(1)
        x = torch.bmm(a, b)
        x = x.reshape(batch_size, -1, 1)
        x = torch.bmm(x, c).unsqueeze(3)
        x = x.reshape(batch_size, self.rank, self.rank, self.rank)

        # fully connected on diagonal of interaction map
        x = torch.stack([x[:, i, i, i] for i in range(self.rank)]).T
        x = self.linear_layer(x)
        x_ = batch['x']
        x = torch.squeeze(x)

        # sigmoid
        x = torch.sigmoid(x)

        if verbose:
            return x, a, b, c
        else:
            return x

    def loss_function(self, x, xhat, a, b, c):
        l1 = torch.sum((x - xhat) ** 2)
        l2 = self.args.regularization * (torch.norm(a) ** 2 + torch.norm(b) ** 2 + torch.norm(c) ** 2)
        loss = l1 + l2
        return loss

    def train(self, dataloader, dataloader_full, logger, epoch):
        # extract parameter
        args = self.args

        # train for one epoch
        for batch in dataloader:
        # for batch in tqdm(dataloader):
            # forward propagation
            xhat, a, b, c = self.forward(batch, verbose=True)
            loss = self.loss_function(batch['x'], xhat, a, b, c)

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()

            # update parameter
            self.optimizer.step()

        # create the description for this epoch
        if epoch == 0 or epoch % self.args.log_epoch == 0:
            with torch.no_grad():
                Xhat = []
                for batch in dataloader_full:
                    xhat = self.forward(batch)
                    Xhat.append(xhat)
                Xhat_scaled = torch.cat(Xhat).reshape(tuple(dataloader_full.dataset.X.shape))
                Xhat = dataloader_full.dataset.scaler.invert_transform(Xhat_scaled)

            # compute the metric
            with torch.no_grad():
                self.stats = logger.summary(dataloader_full, Xhat, Xhat_scaled, loss, epoch)

        # description
        description = 'epoch={} loss={:0.4f} ser={:0.4f}/{:0.4f} ter={:0.4f}/{:0.4f} mae={:0.4f}/{:0.4f}'.format(
                int(epoch), float(loss),
                self.stats['ser'], self.stats['ser_scaled'],
                self.stats['ter'], self.stats['ter_scaled'],
                self.stats['mae'], self.stats['mae_scaled'])
        return description

    def save(self, logger):
        path = logger.log_dir + '.pkl'
        torch.save(self.state_dict(), path)

