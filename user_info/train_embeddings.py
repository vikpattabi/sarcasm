# File responsible for training the subreddit embeddings
import torch
import torch.optim as optim
from user_info import embeddings_model
#from embeddings_model import embeddingModel
import json

import sys
sys.path.append('data')
import read

#from data import read
from read import loadGloveEmbeddings

class subredditEmbeddings:
    def __init__(self, embedding_size, vocabulary, stoi, subreddit_stoi):
        super(subredditEmbeddings, self).__init__()

        # The input is D where D is the word vector size
        self.embedding_size = embedding_size
        self.vocabulary = vocabulary

        self.embeddings = {}
        self.optims = {}

#        self.glove = loadGloveEmbeddings(stoi)
        self.glove = torch.nn.Embedding(num_embeddings=10003, embedding_dim=100).cuda()
        self.glove.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi)).cuda())
        print("Read embeddings")


        self.subreddit_stoi = subreddit_stoi
        self.subreddit_embeddings = torch.nn.Embedding(num_embeddings=1000, embedding_dim=100).cuda()


    def add_subreddit(self, subreddit_name):
        # Use 15 samples
        self.embeddings[subreddit_name] = embeddings_model.embeddingModel(self.embedding_size, self.vocabulary, 15)
        self.optims[subreddit_name] = optim.SGD(self.embeddings[subreddit_name].parameters(), lr=0.001)


    def train_sentence(subreddit, sentence):
        for word in sentence:
            self.embeddings[subreddit].zero_grad()
            out = self.embeddings[subreddit](word)
            out.backward()
            self.optims[subreddit_name].step()

    def train(self, input):
        # input as an iterator
        while True:
            next_sentence = next(input, None)
            if next_sentence == None:
                print('Training done - no more samples.')

            sentence, subredditname = next_sentence
            glove_sentence = [self.glove[w] for w in sentence]

            if subredditname not in self.embeddings.keys():
                self.add_subreddit(subredditname)

            train_sentence(subredditname, glove_sentence)

    def save_to_file(self, filename):
        # Create dict...
        res = {}
        for key in self.embeddings.keys():
            res[key] = self.embeddings[key].embedding

        with open(filename, 'w') as f:
            json.dump(res, f)

    def get(self, subreddit):
        if subreddit in self.embeddings.keys():
            return self.embeddings.embedding
        else:
            print('User ' + str(subreddit) + ' not found.')
            return None
