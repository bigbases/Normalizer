import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hidden_size = 64
        self.output_length = configs.pred_len
        self.num_layers = 2

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

    def forward(self, x):
        """
        x: [batch_size, input_length, num_features]
        return: [batch_size, output_length, num_features]
        """
        _, (hidden, _) = self.encoder(x)  # hidden: [num_layers, batch, hidden_size]
        hidden_last = hidden[-1]       # [batch, hidden_size]

        # Linear projection to future sequence
        out = self.fc(hidden_last)     # [batch, output_length * num_features]
        out = out.view(-1, self.output_length, x.size(-1))  # [batch, output_length, num_features]
        return out
