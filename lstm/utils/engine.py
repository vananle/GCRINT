import pandas as pd
import torch.optim as optim

from .metric import *


class Trainer():
    def __init__(self, model, scaler, lrate, wdecay, clip=3, lr_decay_rate=.97, lossfn='mae'):
        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

        if lossfn == 'mae':
            self.lossfn = mae
        elif lossfn == 'mse':
            self.lossfn = mse
        elif lossfn == 'mae_u':
            self.lossfn = mae_u
        elif lossfn == 'mse_u':
            self.lossfn = mse_u
        else:
            raise ValueError('Loss fn not found!')

    @classmethod
    def from_args(cls, model, scaler, args):
        return cls(model, scaler, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate, lossfn=args.loss_fn)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = torch.nn.functional.pad(input, (1, 0, 0, 0))

        output = self.model(input)  # now, output = [bs, out_seq_len, n]
        predict = self.scaler.inverse_transform(output)

        loss = self.lossfn(predict, real_val)
        rse, mae, mse, mape, rmse = calc_metrics(predict, real_val)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()

    def _eval(self, input, real_val):
        self.model.eval()

        output = self.model(input)  # now, output = [bs, out_seq_len, n]

        predict = self.scaler.inverse_transform(output)

        predict = torch.clamp(predict, min=0., max=10e10)
        loss = self.lossfn(predict, real_val)
        rse, mae, mse, mape, rmse = calc_metrics(predict, real_val)

        return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()

    def test(self, test_loader, model, out_seq_len):
        model.eval()
        outputs = []
        y_real = []
        x_gt = []
        y_gt = []
        for _, batch in enumerate(test_loader):
            x = batch['x']  # [b, seq_x, n, f]
            y = batch['y']  # [b, 1, n]

            preds = model(x)
            preds = self.scaler.inverse_transform(preds)  # [bs, 1, n]
            outputs.append(preds)
            y_real.append(y)
            x_gt.append(batch['x_gt'])  # [b, seq_x, n]
            y_gt.append(batch['y_gt'])  # [b, seq_y, n]

        yhat = torch.cat(outputs, dim=0)
        y_real = torch.cat(y_real, dim=0)
        x_gt = torch.cat(x_gt, dim=0)
        y_gt = torch.cat(y_gt, dim=0)
        test_met = []

        yhat[yhat < 0.0] = 0.0

        for i in range(out_seq_len):
            pred = yhat[:, i, :]
            pred = torch.clamp(pred, min=0., max=10e10)
            real = y_real[:, i, :]
            test_met.append([x.item() for x in calc_metrics(pred, real)])
        test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
        return test_met_df, x_gt, y_gt, y_real, yhat

    def eval(self, val_loader):
        """Run validation."""
        val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = [], [], [], [], [], []
        for _, batch in enumerate(val_loader):
            x = batch['x']  # [b, seq_x, n, f]
            y = batch['y']  # [b, 1, n]

            metrics = self._eval(x, y)
            val_loss.append(metrics[0])
            val_rse.append(metrics[1])
            val_mae.append(metrics[2])
            val_mse.append(metrics[3])
            val_mape.append(metrics[4])
            val_rmse.append(metrics[5])

        return val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse
