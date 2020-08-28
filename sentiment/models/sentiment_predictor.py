import torch
from torch import nn


class SentimentPredictor(nn.Module):

    def __init__(self):
        super(SentimentPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.predictor(x)
        return x


class SentimentPredictorTfIdf(nn.Module):

    def __init__(self, num_words=30185):
        super(SentimentPredictorTfIdf, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(num_words, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.predictor(x)
        return x
