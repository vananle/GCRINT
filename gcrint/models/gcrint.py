import torch


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()


class GraphConvNet(torch.nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = torch.nn.Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = torch.nn.functional.dropout(h, self.dropout, training=self.training)
        return h


class GCRINT(torch.nn.Module):
    def __init__(self, args):
        super(GCRINT, self).__init__()

        self.seq_len = args.seq_len - 2
        self.nSeries = args.nSeries
        # self.nNodes = args.nNodes
        self.lstm_hidden = args.lstm_hidden
        self.gcn_indim = args.gcn_indim
        # self.gcn_hidden = args.gcn_hidden
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.device = args.device
        self.apt_size = args.apt_size
        self.residual_channels = args.residual_channels
        self.verbose = args.verbose

        self.fixed_supports = []

        self.start_conv = torch.nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                          out_channels=self.residual_channels,
                                          kernel_size=(1, 1))
        self.cat_feature_conv = torch.nn.Conv2d(in_channels=self.gcn_indim - 1,
                                                out_channels=self.residual_channels,
                                                kernel_size=(1, 1))

        self.lstm_cell_fw = torch.nn.ModuleList(
            [torch.nn.LSTMCell(input_size=self.residual_channels * self.nSeries,
                               hidden_size=self.lstm_hidden * self.nSeries, bias=True)
             for l in range(self.num_layers)])

        # onnly first layer has backward LSTM
        self.lstm_cell_bw = torch.nn.LSTMCell(input_size=self.residual_channels * self.nSeries,
                                              hidden_size=self.lstm_hidden * self.nSeries, bias=True)

        self.linear_lstm = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.lstm_hidden * self.nSeries, out_features=self.nSeries, bias=True)
             for _ in range(self.num_layers)])

        self.supports_len = len(self.fixed_supports)
        nodevecs = torch.randn(self.nSeries, self.apt_size), torch.randn(self.apt_size, self.nSeries)
        self.supports_len += 1
        self.nodevec1, self.nodevec2 = [torch.nn.Parameter(n.to(self.device), requires_grad=True) for n in nodevecs]

        self.graph_convs = torch.nn.ModuleList(
            [GraphConvNet(self.lstm_hidden, self.residual_channels, self.dropout, support_len=self.supports_len)
             for _ in range(self.num_layers)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.residual_channels) for _ in range(self.num_layers)])

        self.end_conv_1 = torch.nn.Conv2d(self.residual_channels, self.lstm_hidden, (1, 1), bias=True)
        self.end_conv_2 = torch.nn.Conv2d(self.lstm_hidden, self.seq_len, (1, 1), bias=True)

        self.linear_out = torch.nn.Linear(in_features=int(self.seq_len / (2 ** (self.num_layers - 1))), out_features=1)

        self.final_linear_out_1 = torch.nn.Linear(in_features=3 * self.nSeries,
                                                  out_features=128)
        # self.final_linear_out_1 = torch.nn.Linear(in_features=2 * self.nSeries,
        #                                           out_features=128)
        self.final_linear_out_2 = torch.nn.Linear(in_features=128,
                                                  out_features=self.nSeries)

    def lstm_layer(self, x, cell):
        # input x [bs, rc*n, s]
        len = x.size(2)

        h = torch.autograd.Variable(torch.zeros((x.size(0), self.lstm_hidden * self.nSeries)))
        c = torch.autograd.Variable(torch.zeros((x.size(0), self.lstm_hidden * self.nSeries)))

        if torch.cuda.is_available():
            h, c = h.to(self.device), c.to(self.device)

        outputs = []

        for i in range(len):
            _in = x[..., i]
            h, c = cell(_in, (h, c))

            outputs.append(h)

        return outputs

    def feature_concat(self, input_tensor, mask):
        x = torch.stack([input_tensor, mask], dim=1)  # [b, 2, s, n]
        x = x.transpose(2, 3)  # [b, 2, n, s]

        if self.verbose:
            print('input x: ', x.shape)

        f1, f2 = x[:, [0]], x[:, 1:]
        x1 = self.start_conv(f1)
        # x2 = torch.nn.functional.leaky_relu(self.cat_feature_conv(f2))
        x2 = self.cat_feature_conv(f2)
        x = x1 + x2  # [b, rc, n, s]

        return x

    def forward(self, input_tensor, mask, input_tensor_bw, mask_bw, y, ymask):

        # Input: x [b, s, n]
        # w: mask [b, s, n]

        x = self.feature_concat(input_tensor, mask)  # [b, rc, n, s]
        x_bw = self.feature_concat(input_tensor_bw, mask_bw)  # [b, rc, n, s]

        if self.verbose:
            print('After startconv x = ', x.shape)

        # calculate the current adaptive adj matrix once per iteration
        adp = torch.nn.functional.softmax(torch.nn.functional.relu(torch.mm(self.nodevec1, self.nodevec2)),
                                          dim=1)  # the learnable adj matrix
        adjacency_matrices = self.fixed_supports + [adp]

        if self.verbose:
            print('Adjmx: ', adjacency_matrices[0].shape)

        in_y = y
        outputs = 0
        for l in range(self.num_layers):

            in_lstm = x.reshape(x.size(0), -1, x.size(3))  # [b, rc*n, s]
            len = in_lstm.size(2)

            if self.verbose:
                print('layer {} input = {}'.format(l, in_lstm.shape))

            gcn_in = self.lstm_layer(in_lstm, self.lstm_cell_fw[l])  # fw lstm
            gcn_in = torch.stack(gcn_in, dim=-1)  # [b, n*h, len]

            if l == 0:
                in_lstm_bw = x_bw.reshape(x_bw.size(0), -1, x_bw.size(3))  # [b, rc*n, s]

                gcn_in_bw = self.lstm_layer(in_lstm_bw, self.lstm_cell_bw)  # bw lstm layer
                gcn_in_bw = torch.stack(gcn_in_bw, dim=-1)  # [b, n*h, len]     bw outputs
                gcn_in_bw = torch.flip(gcn_in_bw, dims=[-1])  # flip bw output

                # c_loss = torch.abs(gcn_in - gcn_in_bw).mean() * 1e-1            # consistency loss

                gcn_in = (gcn_in + gcn_in_bw) / 2.0  # combine 2 outputs

            gcn_in = gcn_in.reshape(-1, self.lstm_hidden, self.nSeries, len)

            gcn_in = torch.tanh(gcn_in)

            if self.verbose:
                print('gcn in = ', gcn_in.shape)

            graph_out = self.graph_convs[l](gcn_in, adjacency_matrices)  # [b, rc, n, len]
            if self.verbose:
                print('gcn out = ', graph_out.shape)

            try:
                outputs = outputs[:, :, :, -graph_out.size(3):]
            except:
                outputs = 0
            outputs = graph_out + outputs

            if self.verbose:
                print('skip outputs = ', outputs.shape)

            x = graph_out
            x = x[..., 0:len:2]  # [b, rc, n, s/2 ]
            # in_y = in_y[..., 0:len:2]  # [b, rc, n, s/2 ]
            x = self.bn[l](x)

            if self.verbose:
                print('---------------------------------')

        if self.verbose:
            print('final skip outputs = ', outputs.shape)

        # outputs = torch.nn.functional.relu(outputs)  # [b, gcn_hidden, n, seq/L]
        outputs = self.end_conv_1(outputs)  # [b, h', n, s/L]
        # outputs = torch.nn.functional.relu(self.end_conv_1(outputs))  # [b, h', n, s/L]
        outputs = self.end_conv_2(outputs)  # [b, s, n, s/L]
        if self.verbose:
            print('outputs end_conv = ', outputs.shape)

        outputs = self.linear_out(outputs)  # [b, s, n, 1]
        outputs = outputs.squeeze(dim=-1)  # [b, s, n]
        # outputs = outputs + y
        outputs = torch.cat([outputs, y, ymask], dim=-1)
        outputs = self.final_linear_out_1(outputs)
        outputs = self.final_linear_out_2(outputs)

        if self.verbose:
            print('final outputs = ', outputs.shape)
            print('*******************************************************************')

        return outputs
