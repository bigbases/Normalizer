import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):

        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        # change trend from value-based to change-based
        last_value = trend_init[:, :, -1:].clone()
        trend_change = torch.zeros_like(trend_init)
        trend_change[:, :, 1:] = trend_init[:, :, 1:] - trend_init[:, :, :-1]
        trend_init = trend_change
        
        # trend normalization
        trend_std = torch.std(trend_init, dim=2, keepdim=True)
        trend_init = trend_init / (trend_std + 1e-5)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init).permute(0, 2, 1)
        trend_output = self.Linear_Trend(trend_init)
        
        # trend denormalization
        trend_output = trend_output * (trend_std + 1e-5)

        trend_output = trend_output.cumsum(dim=2) + last_value
        trend_output = trend_output.permute(0, 2, 1)
        
        x = seasonal_output + trend_output
        return x, seasonal_output, trend_output
