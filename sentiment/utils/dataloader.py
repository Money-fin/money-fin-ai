import numpy as np
import pathlib
import re
import pickle
from functools import partial
from sklearn.model_selection import train_test_split
from konlpy.tag import Kkma
from gensim.models.word2vec import Word2Vec

DATA_PATH = "../../../data/moneyfin/text/"
INDEXING_DATA_PATH = "../../../data/moneyfin/indexing"
EMAIL_PATTERN = r"[a-zA-Z0-9]+@[a-zA-Z0-9]+.[a-zA-Z]+"

class DataLoader():

    def __init__(self, batch_size, mode="train"):
        self.batch_size = batch_size
        self.data_list = self._load_data_list()
        self.kkma = Kkma()
        # self.word_dict = dict()

        self.wv = Word2Vec.load("embedding/ko.bin")

        trainset, testset = train_test_split(self.data_list, test_size=0.2, random_state=9)
        trainset, validset = train_test_split(trainset, test_size=0.2, random_state=9)

        if mode == "train":
            self.data_list = trainset
        elif mode == "valid":
            self.data_list = validset
        elif mode == "test":
            self.data_list = testset
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.index = 0

    def __len__(self):
        return int(np.ceil(len(self.data_list) / self.batch_size))

    def __next__(self):
        if self.index >= len(self.data_list):
            self.index = 0

        article = []
        label = -1

        while len(article) == 0:
            index = self.index
            self.index += 1
            article, label = self._read_one_article(self.data_list[index])

        tokens = self._preprocess(article)
        # tokens = article

        return tokens, label

    def __iter__(self):
        return self

    def next_batch(self):
        np.random.shuffle(self.data_list)

        x_batch = []
        y_batch = []
        self.index = 0

        while self.index < len(self.data_list):
            if len(x_batch) < self.batch_size:
                x, y = next(self)
                x_batch.append(x)
                y_batch.append(y)
            else:
                yield x_batch, y_batch
                x_batch = []
                y_batch = []

        if len(x_batch) > 0:
            yield x_batch, y_batch

    def _load_data_list(self):
        data_list = []

        for p in pathlib.Path(DATA_PATH).glob("*.txt"):
            data_list.append(str(p))

        return data_list

    def _read_one_article(self, path):
        lines = []

        with open(path, "r", encoding="utf8") as f:
            label = int(f.readline().strip())

            for line in f:
                line = line.replace("\n", "").strip()

                if re.search(EMAIL_PATTERN, line) or line.startswith("▶"):
                    break

                if len(line) > 0:
                    splited = line.split(r".[\s]")
                    for sen in splited:
                        if len(sen.strip()) > 0:
                            lines.append(sen.strip())

        return lines, label

    def _preprocess(self, lines):
        tokens = []

        def map_fn(noun, wv):
            if noun in wv.wv.vocab.keys():
                return wv.wv.vocab.get(noun).index
            else:
                return -1

        for line in lines:

            # for noun in self.kkma.nouns(line):
            #     if noun not in self.word_dict.keys():
            #         self.word_dict[noun] = len(self.word_dict)

            #     tokens[-1].append(int(self.word_dict[noun]))

            nouns = self.kkma.nouns(line)
            nouns_indices = map(partial(map_fn, wv=self.wv), nouns)
            nouns_indices = list(filter(lambda item: item != -1, nouns_indices))
            tokens.append(nouns_indices)

            # for noun in self.kkma.nouns(line):
            #     if noun in self.wv.wv.vocab.keys():
            #         tokens[-1].append(self.wv.wv.vocab.get(noun).index)

            if len(tokens[-1]) == 0:
                del tokens[-1]

        return tokens


class TfIdfDataLoader():

    def __init__(self, batch_size, mode="train"):
        self.batch_size = batch_size
        self.data_list = self._load_data_list()
        self.kkma = Kkma()
        # self.word_dict = dict()

        self.wv = Word2Vec.load("embedding/ko.bin")

        trainset, testset = train_test_split(self.data_list, test_size=0.2, random_state=9)
        trainset, validset = train_test_split(trainset, test_size=0.2, random_state=9)

        if mode == "train":
            self.data_list = trainset
        elif mode == "valid":
            self.data_list = validset
        elif mode == "test":
            self.data_list = testset
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.index = 11528

    def __len__(self):
        return int(np.ceil(len(self.data_list) / self.batch_size))

    def __next__(self):
        if self.index >= len(self.data_list):
            self.index = 0

        article = []
        label = -1

        while len(article) == 0:
            index = self.index
            self.index += 1
            article, label = self._read_one_article(self.data_list[index])
        # tokens = article

        return article, label

    def __iter__(self):
        return self

    def next_batch(self):
        np.random.shuffle(self.data_list)

        x_batch = []
        y_batch = []
        self.index = 0

        while self.index < len(self.data_list):
            if len(x_batch) < self.batch_size:
                x, y = next(self)
                x_batch.append(x)
                y_batch.append(y)
            else:
                yield x_batch, y_batch
                x_batch = []
                y_batch = []

        if len(x_batch) > 0:
            yield x_batch, y_batch

    def _load_data_list(self):
        data_list = []

        for p in pathlib.Path(INDEXING_DATA_PATH).glob("*_tfidf.bin"):
            data_list.append(str(p))

        with open(f"{INDEXING_DATA_PATH}/y.bin", "rb") as f:
            labels = pickle.loads(f.read())

        data_list = list(zip(data_list, labels))

        return data_list

    def _read_one_article(self, path):
        path, label = path

        with open(path, "rb") as f:
            lines = pickle.loads(f.read())

        return lines, label



