import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

class NormLosses(torch.nn.Module):
    def __init__(self, args, station=None):
        super(NormLosses, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss()
        self.decomp = series_decomp(args.kernel_len)
        if args.use_norm == 'DDN':
            self.station = station

    # SAN
    def san_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    # DDN
    def sliding_loss(self, y, statistics_pred):
        _, (mean, std) = self.station.norm(y.transpose(-1, -2), False)
        station_ture = torch.cat([mean, std], dim=1).transpose(-1, -2)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    # TP
    def trend_loss(self, y, statistics_pred):
        trend_pred = statistics_pred[-1]
        _, trend_true = self.decomp(y)
        loss = self.criterion(trend_pred, trend_true)
        return loss

    # LightTrend
    def lt_loss(self, y, statistics_pred):
        trend_pred = statistics_pred
        _, trend_true = self.decomp(y)
        loss = self.criterion(trend_pred, trend_true)
        return loss
    
    def forward(self, y, statistics_pred):
        if self.args.use_norm == 'san':
            loss = self.san_loss(y, statistics_pred)
        elif self.args.use_norm == 'ddn':
            loss = self.sliding_loss(y, statistics_pred)
        elif self.args.use_norm == 'tp':
            loss = self.trend_loss(y, statistics_pred)
        elif self.args.use_norm == 'lt':
            loss = self.lt_loss(y, statistics_pred)
        else:
            loss = torch.tensor(0.0)
        return loss