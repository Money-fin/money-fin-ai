import torch
from torch import nn
import random


class MINE(nn.Module):

    def __init__(self):
        super(MINE, self).__init__()

        self.sentence_kernel = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh()
        )

        self.context_kernel = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh()
        )

    def forward(self, sentences, contexts):
        for sen, con in zip(sentences, contexts):
            N, T, F = sen.size()

            sen = sen.view(N*T, -1)
            con = con.repeat(T, 1)

            sen_embedded = self.sentence_kernel(sen)
            con_embedded = self.context_kernel(con)

            bilinear = torch.mm(con_embedded, sen_embedded.t())
            exp_bilinear = torch.exp(bilinear)
            # diag_exp_bilinear = exp_bilinear * torch.eye(N).to(sen.device)

            positive = torch.trace(exp_bilinear)

        random.shuffle(sentences)

        for sen, con in zip(sentences, contexts):
            N, T, F = sen.size()

            sen = sen.view(N*T, -1)
            con = con.repeat(T, 1)

            sen_embedded = self.sentence_kernel(sen)
            con_embedded = self.context_kernel(con)

            bilinear = torch.mm(con_embedded, sen_embedded.t())
            exp_bilinear = torch.exp(bilinear)
            # non_diag_exp_bilinear = exp_bilinear * (1 - torch.eye(N).to(sen.device))

            negative = torch.sum(exp_bilinear, dim=1)

        # print(positive)
        # print(negative)

        cpc = torch.mean(torch.log(positive)) - torch.mean(torch.log(negative))
        return cpc
