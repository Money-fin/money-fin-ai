import torch
from torch import nn


class SentimentPredictor(nn.Module):

    def __init__(self):
        super(SentimentPredictor, self).__init__()

        # self.features = nn.LSTM(
        #     input_size=768,
        #     hidden_size=32,
        #     num_layers=1,
        #     batch_first=True,
        # )

        # self.predictor = nn.Sequential(
        #     nn.Linear(768, 1),
        #     nn.Sigmoid()
        # )

        # self.predictor = nn.Sequential(
        #     nn.Linear(768, 32),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),

        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )

        self.predictor = nn.Sequential(
            nn.Linear(768, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x, _ = self.features(x)
        # pred = self.predictor(x[:, -1])
        pred = self.predictor(x)
        return pred


class SentimentPredictorWithContents(nn.Module):

    def __init__(self, num_words=30185):
        super(SentimentPredictorWithContents, self).__init__()

        self.features = nn.LSTM(
            input_size=768,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )

        self.predictor = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.features(x)
        x = self.predictor(x[:, -1])
        return x
