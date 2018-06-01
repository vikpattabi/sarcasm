# File responsible for training the user embeddings
import torch
import torch.optim as optim
from embeddings_model import embeddingModel
import json

import sys
sys.path.append('../data')
from read import loadGloveEmbeddings

class userEmbeddings:
    def __init__(self, embedding_size, vocabulary, stoi):
        super(userEmbeddinds, self).__init__()

        # The input is D where D is the word vector size
        self.embedding_size = embedding_size
        self.vocabulary = vocabulary

        self.embeddings = {}
        self.optims = {}

        self.glove = loadGloveEmbeddings(stoi)

    def add_user(self, user_name):
        # Use 15 samples
        self.embeddings[user_name] = embeddingModel(self.embedding_size, self.vocabulary, 15)
        self.optims[user_name] = optim.SGD(self.embeddings[user_name].parameters(), lr=0.001)


    def train_sentence(user, sentence):
        for word in sentence:
            self.embeddings[user].zero_grad()
            out = self.embeddings[user](word)
            out.backward()
            self.optims[user_name].step()

    def train(self, input):
        # input as an iterator
        while True:
            next_sentence = next(input, None)
            if next_sentence == None:
                print('Training done - no more samples.')

            sentence, username = next_sentence
            glove_sentence = [self.glove[w] for w in sentence]

            if username not in self.embeddings.keys():
                self.add_user(username)

            train_sentence(username, glove_sentence)

    def save_to_file(self, filename):
        # Create dict...
        res = {}
        for key in self.embeddings.keys():
            res[key] = self.embeddings[key].embedding

        with open(filename, 'w') as f:
            json.dump(res, f)
