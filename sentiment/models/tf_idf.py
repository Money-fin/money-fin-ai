import numpy as np
import torch
import copy
import pickle
from  torch import nn


class TfIdf():

    def __init__(self, num_words=30185, num_docs=18000):
        self.df_scores = dict()
        self.num_words = num_words
        self.num_docs = num_docs

    def fit(self, doc):
        bow = dict()

        for sentence in doc:
            for word in sentence:
                if word in bow.keys():
                    bow[word] += 1
                else:
                    bow[word] = 1

        for k, v in bow.items():
            if k in self.df_scores.keys():
                self.df_scores[k] += 1
            else:
                self.df_scores[k] = 1

    def predict(self, doc):
        bow = dict()

        for sentence in doc:
            for word in sentence:
                if word in bow.keys():
                    bow[word] += 1
                else:
                    bow[word] = 1

        tf_idf_scores = np.zeros((self.num_words,))
        for k, v in bow.items():
            if k in self.df_scores.keys():
                tf_idf_scores[k] = bow[k] * np.log(self.num_docs/self.df_scores[k])

        return tf_idf_scores

    def save(self, path):
        with open(path, "wb") as f:
            f.write(pickle.dumps(self.df_scores))

    def load(self, path):
        with open(path, "rb") as f:
            self.df_scores = pickle.loads(f.read())
        
