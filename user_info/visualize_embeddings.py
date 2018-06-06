# File responsible for training the subreddit embeddings
import torch
from user_info import embeddings_model
from user_info import tsne

#from embeddings_model import embeddingModel
import json

import sys
sys.path.append('data')
import read

#from data import read
from read import loadGloveEmbeddings

import pylab


# Trains subreddit embeddings for one epoch.
# normalizeEmbeddings: from time to time normalize the L2 norm of each subreddit embedding to one
# I haven't really tuned these parameters very much.
def trainSubredditEmbeddings(normalizeEmbeddings = False, learning_rate = 0.2, learning_rate_decay = 0.99, batchSize = 10, number_of_negative_samples = 15, loadFromFile=False):
     itos, stoi = read.createVocabulary(vocab_size=10000) # restrict to 10000 most frequent words
     print("Read dictionary")
     
     #itos_subreddits, stoi_subreddits = read.readUserDictionary()
     itos_subreddits, stoi_subreddits = read.readSubredditDictionary()
     
     
     subreddit_index = read.keys.index("subreddit")
     
     
     def subredditDataIterator():
        data = read.dataIterator(doTokenization=False, printProblems=False)
        counter = 0
        for sentence in data:
           if sentence[0] == "1":
                continue
           subreddit_id = stoi_subreddits.get(sentence[subreddit_index], None)
           if subreddit_id is not None:
             counter += 1
             if counter % 10000 == 0:
                print(counter)
             read.tokenize(sentence) # this step is the bottleneck
             tokens = sentence[1]
             for token in tokens:
                token = token.lower()
                token_id = stoi.get(token, None)
                if token_id is not None and token_id < 10000:
                    yield (token_id, subreddit_id)
     
     import numpy as np
     import torch
     import torch.nn
     
     subredditTrainingData = subredditDataIterator()
     
     unigram_probabilities = read.getUnigramProbabilities(vocab_size=10000)
     
     subreddit_embeddings = torch.nn.Embedding(num_embeddings=1000, embedding_dim=100).cuda()
     
     
     
     glove = torch.nn.Embedding(num_embeddings=10000, embedding_dim=100).cuda()
     glove.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi, offset=0, itos=itos)).cuda())
     
     norm = torch.norm(glove.weight.data, p=2, dim=1).unsqueeze(1)
     
     glove.weight.data = glove.weight.data.div(norm.expand_as(glove.weight.data))
     
     print("Read embeddings")
     
     if loadFromFile:
        load = torch.load( "data/checkpoints/subreddit_embeddings.pth")
   #{"subreddit_embeddings" : subreddit_embeddings.state_dict(), "glove_embeddings" : glove.state_dict()},
        subreddit_embeddings.load_state_dict(load["subreddit_embeddings"])
        glove.load_state_dict(load["glove_embeddings"])
     
     def printClosestNeighbors():
        subreddits = subreddit_embeddings.weight.data[:20].view(20, 1, 1, 100) # 20 x 100
        words = glove.weight.data[:5000].view(1, 5000, 100, 1)
        scores = torch.matmul(subreddits, words) # 20 x 10000 x 1 x 1
        scores = scores.view(20, 5000)
        best = torch.topk(scores, k=10)[1].data.cpu().numpy()
        for i in range(20):
           print("------------")
           print(itos_subreddits[i])
           print([itos[j] for j in best[i]])
     
     import matplotlib.pyplot as plt

     def printClosestNeighborsSubreddits():
        subreddits = subreddit_embeddings.weight.data[:20].view(20, 1, 1, 100) # 20 x 100
        subredditsTarget = subreddit_embeddings.weight.data[:100].view(1, 100, 100, 1)
        scores = torch.matmul(subreddits, subredditsTarget) # 20 x 10000 x 1 x 1
        scores = scores.view(20, 100)
        best = torch.topk(scores, k=10)[1].data.cpu().numpy()
        for i in range(20):
           print("------------")
           print(itos_subreddits[i])
           print([itos_subreddits[j] for j in best[i] if j != i])
        first_n = 50
        X = subreddit_embeddings.weight.data[:first_n].cpu().numpy()
        Y = tsne.tsne(X=X, no_dims=2, initial_dims=10, perplexity=30.0)
        labels = itos_subreddits[:first_n]
        fig, ax = plt.subplots()
        for i in range(first_n):
           ax.annotate(itos_subreddits[i], (Y[i,0], Y[i,1]))
        ax.scatter(Y[:, 0], Y[:, 1], 20, [5.0 for _ in range(first_n)])

        plt.show()
   
        #pylab.scatter(Y[:, 0], Y[:, 1], 20, [5.0 for _ in range(50)])
        #pylab.show()
     
     
#     runningAverageLoss = 1
#     counterTraining = 0
#     while True:
#         counterTraining += 1
#         batch = [next(subredditTrainingData) for _ in range(batchSize)]
#         tokens = [x[0] for x in batch]
#         subreddits = [x[1] for x in batch]
#         negative_samples = np.random.choice(10000, size=(batchSize*number_of_negative_samples), p=unigram_probabilities)
#         positive_tokens = glove(torch.LongTensor(tokens).cuda()) # (batchSize, 100)
#         negative_tokens = glove(torch.LongTensor(negative_samples).cuda()) # (batchSize * negSamples, 100)
#         subreddit_embedded = subreddit_embeddings(torch.LongTensor(subreddits).cuda()) # (batchSize, 100)
#         dotProductPositive = torch.bmm(subreddit_embedded.unsqueeze(1), positive_tokens.unsqueeze(2)).unsqueeze(1) # (batchSize, 1, 1)
#     
#         first =subreddit_embedded.unsqueeze(1).unsqueeze(2)
#         second = negative_tokens.view(batchSize, number_of_negative_samples, 100, 1)
#         dotProductNegative = torch.matmul(first, second)
#         dotProductNegative = dotProductNegative.view(batchSize, number_of_negative_samples, 1)
#     
#         loss = torch.nn.ReLU()(1-dotProductPositive+dotProductNegative)
#         meanLoss = loss.mean()
##         meanLoss.backward()
#     
#         runningAverageLoss = 0.999 * runningAverageLoss + (1-0.999) * meanLoss.data.cpu().numpy()
#         if counterTraining % 1000 == 0:
#            print(runningAverageLoss)
#            if counterTraining % 50000 == 0:
     printClosestNeighbors()
     printClosestNeighborsSubreddits()
    
      #         if normalizeEmbeddings:
      #            norm = torch.norm(subreddit_embeddings.weight.data, p=2, dim=1).unsqueeze(1).detach()
      #            subreddit_embeddings.weight.data = subreddit_embeddings.weight.data.div(norm.expand_as(subreddit_embeddings.weight.data))
     
      #      if counterTraining % 100000 == 0:
      #         learning_rate *= learning_rate_decay
      #         print("Decaying learning rate")
      #         print(learning_rate)
      #         print("Storing embeddings in file")
    
     
     
     

