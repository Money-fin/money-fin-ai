from embedding import Embedding
import torch.optim as optim
import torch.nn as nn
import csv
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import unicodedata
import random
from tqdm import tqdm



def plot_losses(losses):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    axes.plot(losses)
    axes.set_title("train loss")
    axes.set_xlabel("epochs")
    axes.set_ylabel("loss")

    plt.tight_layout()
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, input_):
        o = self.fc(input_)
        o = self.relu(o)
        o = self.fc2(o)
        o = self.sig(o)
        return o

LABEL_MAP = {
    # filename: model_input
    'chemistry': "화학",
    "bio": "생명",
    "IT": "정보통신",
}

def create_dataset(target, *others):
    
    dataset = []
    n_target = len(target)
    
    X = list(range(n_target))
    dataset += target
    Y = [1 for _ in range(n_target)]
    
    n_X = len(X)
    
    _others = []
    for other in others:
        _others += other
        dataset += other
        
    import random
    random.shuffle(_others)
        
    n_max = min(n_X, len(_others))
    X = X[:n_max]
    Y = Y[:n_max]
    
    X = X + list(range(n_X, n_X + len(_others)))
    Y = Y + [0 for _ in _others]
    X = X[:2 * n_max]
    Y = Y[:2 * n_max]
        
    assert len(X) == len(Y)
    
    train, test = train_test_split(X, train_size=0.9, test_size=0.1, stratify=Y)
    train = [dataset[i] for i in train]
    test = [dataset[i] for i in test]
    
    counter_train = Counter([x[1] for x in train])
    counter_test = Counter([x[1] for x in test])
    return train, test  

def create_input(text, sector, label):
    x = Embedding.get_classification_vector(text, sector)
    y = 1 if sector == label else 0
    return x, [y]


def load_data(filename):
    data = []
    with open(f'./classification/{filename}.csv') as f:
        rows = csv.reader(f)
        for row in rows:
            data.append((row[1], filename))
    return data  # [(text, label)]



class MFCModel(object):
    """MoneyFine classification model"""
    LABEL_MAP = {
        # filename: model_input
        'chemistry': "화학",
        "bio": "생명",
        "it": "정보통신",
    }
    PATH = "./classification/models/5."
    
    
    def __init__(self):
        self.net = Net()
        self.opt = optim.Adam(self.net.parameters(), lr=0.005, betas=(0.9, 0.999))
        self.criterion = nn.BCELoss()
        
    def create_dataset(self):
        train_set = {}
        for k, v in self.LABEL_MAP.items():
            train_set[k] = load_data(k)
            
        return train_set
        
    def inference(self, text):
        
        result_map = {}
        for k, v in self.LABEL_MAP.items():
            tv = Embedding.get_classification_vector(text, v)
            pred = self.net(tv.reshape(1, -1))
            result_map[v] = float(pred)

        return result_map
    
    def __call__(self, text):
        return self.inference(text)
    
#     def train(self):
#         data_map = {}
#         for k, v in self.LABEL_MAP.items():
#             data = load_data(k)
#             data_map[k] = data
        
        
#         target_data = data_map[target]
#         other_data = []
#         for k, v in data_map.items():
#             if k != target:
#                 other_data.append(data_map[k])
        
#         train, test = create_dataset(target_data, *other_data)
    
    def _random(self):
        return bool(random.choice([0, 1]))

    def train(self):  # LABEL_MAP key
        train_set = self.create_dataset()
        
        data = []
        losses = []
        for i in range(20):
            for k, v in train_set.items():  # k, filename, v, data
                xs = []
                ys = []
                
                for each in tqdm(v):
                    filename = k
                    text = each[0]
                    label = each[1]

                    other_labels = [k for k ,v in self.LABEL_MAP.items() if k != label]
                    
                    sector = None
                    if self._random():
                        sector = self.LABEL_MAP[label]
                    else:
                        sector = self.LABEL_MAP[random.choice(other_labels)]

                    x, y = create_input(text, sector, self.LABEL_MAP[label])
                    xs.append(x)
                    ys.append(y)
                
                x = torch.stack(xs)
                y = torch.FloatTensor(ys)
                
                
                loss = self._train_model(x, y, self.opt, self.criterion, batch_size=20)
                losses.append([float(each) for each in loss])
#                     print(f'text={text}, label={label}, sector={sector}, y={y[0][0]}, loss={float(loss[0])}')

#                 plot_losses(losses)
            self.save(i)
        return losses
            
    def _train_model(self, X, Y, opt, criterion, batch_size):
        # X: [batch_size, 768]
        # Y: [batch_size, 1]
        net = self.net
        net.train()
        losses = []
        for beg_i in range(0, X.size(0), batch_size):
            x_batch = X[beg_i:beg_i + batch_size, :]
            y_batch = Y[beg_i:beg_i + batch_size, :]
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)

            opt.zero_grad()
            # (1) Forward
            y_hat = net(x_batch)
            # (2) Compute diff
            loss = criterion(y_hat, y_batch)
            # (3) Compute gradients
            loss.backward()
            # (4) update weights
            opt.step()
            losses.append(loss.data.numpy())
        print(losses)
        return losses
    
    def save(self, surfix):
        torch.save(self.net.state_dict(), self.PATH + str(surfix))
    
    def load(self, surfix):
        print(f"model load from {self.PATH + str(surfix)}")
        self.net.load_state_dict(torch.load(self.PATH + str(surfix)))
