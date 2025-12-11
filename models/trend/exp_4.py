import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hidden_size = 256
        self.output_length = configs.pred_len
        self.num_layers = 2
        
        # Decompsition Kernel Size
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout if self.num_layers > 1 else 0.0
        )

        # Linear Decoder
        self.fc = nn.Linear(self.hidden_size, self.output_length * configs.enc_in)
        
        self.Linear_Seasonal = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x):
        """
        x: [batch_size, input_length, num_features]
        return: [batch_size, output_length, num_features]
        """
        
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        # change trend from value-based to change-based
        last_value = trend_init[:, :, -1:].clone()
        trend_change = torch.zeros_like(trend_init)
        trend_change[:, :, 1:] = trend_init[:, :, 1:] - trend_init[:, :, :-1]
        trend_init = trend_change.permute(0, 2, 1)
        
        _, (hidden, _) = self.encoder(trend_init)  # hidden: [num_layers, batch, hidden_size]
        hidden_last = hidden[-1]       # [batch, hidden_size]

        # Linear projection to future sequence
        trend_output = self.fc(hidden_last)     # [batch, output_length * num_features]
        trend_output = trend_output.view(-1, self.output_length, x.size(-1))  # [batch, output_length, num_features]
        trend_output = trend_output.cumsum(dim=1) + last_value.permute(0, 2, 1)
        
        # Seasonal
        seasonal_output = self.Linear_Seasonal(seasonal_init).permute(0, 2, 1)
        
        x = seasonal_output + trend_output
        return x, seasonal_output, trend_output
