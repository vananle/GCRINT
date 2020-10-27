import torch.optim as optim

from .metric import *
from .recover import *


# Imputation Engine
class ImpEngine():
    def __init__(self, model, scaler, lrate, wdecay, clip=3, lr_decay_rate=.97, args=None):
        self.args = args
        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    @classmethod
    def from_args(cls, model, scaler, args):
        return cls(model, scaler, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate, args=args)

    # def train(self, input, real_val):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     # input = torch.nn.functional.pad(input, (1, 0, 0, 0))
    #
    #     output = self.model(input)  # now, output = [bs, seq_y, n]
    #     predict = self.scaler.inverse_transform(output)
    #
    #     loss = self.lossfn(predict, real_val)
    #     rse, mae, mse, mape, rmse = calc_metrics(predict, real_val)
    #     loss.backward()
    #
    #     if self.clip is not None:
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
    #     self.optimizer.step()
    #     return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()
    #
    # def _eval(self, input, real_val):
    #     self.model.eval()
    #
    #     output = self.model(input)  # now, output = [bs, seq_y, n]
    #
    #     predict = self.scaler.inverse_transform(output)
    #
    #     predict = torch.clamp(predict, min=0., max=10e10)
    #     loss = self.lossfn(predict, real_val)
    #     rse, mae, mse, mape, rmse = calc_metrics(predict, real_val)
    #
    #     return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()
    #
    # def test(self, test_loader, model, out_seq_len):
    #     model.eval()
    #     outputs = []
    #     y_real = []
    #     x_gt = []
    #     y_gt = []
    #     for _, batch in enumerate(test_loader):
    #         x = batch['x']  # [b, seq_x, n, f]
    #         y = batch['y']  # [b, seq_y, n]
    #
    #         preds = model(x)
    #         preds = self.scaler.inverse_transform(preds)  # [bs, seq_y, n]
    #         outputs.append(preds)
    #         y_real.append(y)
    #         x_gt.append(batch['x_gt'])
    #         y_gt.append(batch['y_gt'])
    #
    #     yhat = torch.cat(outputs, dim=0)
    #     y_real = torch.cat(y_real, dim=0)
    #     x_gt = torch.cat(x_gt, dim=0)
    #     y_gt = torch.cat(y_gt, dim=0)
    #     test_met = []
    #
    #     yhat[yhat < 0.0] = 0.0
    #
    #     for i in range(out_seq_len):
    #         pred = yhat[:, i, :]
    #         pred = torch.clamp(pred, min=0., max=10e10)
    #         real = y_real[:, i, :]
    #         test_met.append([x.item() for x in calc_metrics(pred, real)])
    #     test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
    #     return test_met_df, x_gt, y_gt, y_real, yhat
    #
    # def eval(self, val_loader):
    #     """Run validation."""
    #     val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = [], [], [], [], [], []
    #     for _, batch in enumerate(val_loader):
    #         x = batch['x']  # [b, seq_x, n, f]
    #         y = batch['y']  # [b, seq_y, n]
    #
    #         metrics = self._eval(x, y)
    #         val_loss.append(metrics[0])
    #         val_rse.append(metrics[1])
    #         val_mae.append(metrics[2])
    #         val_mse.append(metrics[3])
    #         val_mape.append(metrics[4])
    #         val_rmse.append(metrics[5])
    #
    #     return val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse

    def train(self, dataloader):
        losses = []
        for i, batch in enumerate(dataloader):
            self.model.train()

            # forward propagation
            batch['x_hat'] = self.model(batch['x'], batch['w'], batch['xi'], batch['wi'], batch['y'], batch['wy'])
            loss = partial_mae(batch['y'], batch['x_hat'], batch['wy'])
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        return loss

    def validation(self, dataloader, val_dataloader):
        self.model.eval()

        X = dataloader.dataset.X
        W = dataloader.dataset.W
        Wval = val_dataloader.dataset.W

        Xhat = self.recovery(val_dataloader)

        _rse, _mae, _mape, _mse, _rmse = calculate_metrics(X, Xhat, W - Wval)

        val_loss = _mse
        metrics = (val_loss, _rse, _mae, _mape, _mse, _rmse)
        return Xhat, metrics

    def imputation(self, dataloader):
        self.model.eval()

        X = dataloader.dataset.X
        W = dataloader.dataset.W
        Xhat = self.recovery(dataloader)

        X_linear_imp = dataloader.dataset.X_linear_imp

        _rse, _mae, _mape, _mse, _rmse = calculate_metrics(X, Xhat, W)
        metrics = (_rse, _mae, _mape, _mse, _rmse)

        _rse_li, _mae_li, _mape_li, _mse_li, _rmse_li = calculate_metrics(X, X_linear_imp, W)
        metrics_li = (_rse_li, _mae_li, _mape_li, _mse_li, _rmse_li)

        return Xhat, metrics, metrics_li

    def recovery(self, dataloader):
        # batch = extract_all_sub_series(dataloader.dataset)
        # batch['xhat'] = self.model(batch['x'], batch['w'])
        # Xhat = build_xhat(dataloader.dataset, batch)
        batch = extract_all_sub_series_overlap(dataloader.dataset)

        batch_xhat = []
        for i in range(0, batch['x'].shape[0], self.args.val_batch_size):
            _stop = i + self.args.val_batch_size if i + self.args.val_batch_size <= batch['x'].shape[0] else \
                batch['x'].shape[0]

            x = batch['x'][i: _stop]
            w = batch['w'][i: _stop]
            xi = batch['xi'][i: _stop]
            wi = batch['wi'][i: _stop]
            y = batch['y'][i: _stop]
            wy = batch['wy'][i: _stop]

            xhat = self.model(x, w, xi, wi, y, wy)
            batch_xhat.append(xhat)

        batch_xhat = torch.cat(batch_xhat, dim=0)
        Xhat = build_xhat_overlap(dataloader.dataset, batch_xhat)
        Xhat[Xhat < 0.0] = 0.0

        return Xhat
