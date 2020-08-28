import torch
from torch import nn
from gensim.models.word2vec import Word2Vec


class SentenceEncoder(nn.Module):

    def __init__(self):
        super(SentenceEncoder, self).__init__()

        wv = Word2Vec.load("embedding/ko.bin")
        self.embedding = nn.Embedding(wv.wv.vectors.shape[0], wv.wv.vectors.shape[1])
        self.embedding.weight = nn.Parameter(torch.FloatTensor(wv.wv.vectors))
        self.embedding.requires_grad_(False)

        self.encoder = nn.LSTM(
            input_size=200,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        N, T, F = x.size()
        x = x.view(N*T, F)
        x = self.embedding(x)
        x = x.view(N, T, -1)

        outputs, _ = self.encoder(x)
        return outputs[:, -1]
