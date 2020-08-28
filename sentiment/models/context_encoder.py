import torch
from torch import nn


class ContextEncoder(nn.Module):

    def __init__(self):
        super(ContextEncoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=32,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        outputs, _ = self.encoder(x)
        return outputs[:, -1]
