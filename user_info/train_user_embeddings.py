# File responsible for training the user embeddings
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

# expecting embeddings as a numpy array
def save_to_file(itos_users, embeddings):
    with open("data/embeddings/user_embeddings.txt", "w") as outFile:
       for i in range(len(embeddings)):
          line = [itos_users[i]] + [str(x) for x in embeddings[i].tolist()]
          outFile.write(("\t".join(line))+"\n")




# Trains user embeddings for one epoch.
# normalizeEmbeddings: from time to time normalize the L2 norm of each user embedding to one
# I haven't really tuned these parameters very much.
def trainUserEmbeddings(normalizeEmbeddings = False, learning_rate = 0.2, learning_rate_decay = 0.99, batchSize = 10, number_of_negative_samples = 15):
     itos, stoi = read.createVocabulary(vocab_size=10000) # restrict to 10000 most frequent words
     print("Read dictionary")
     
     #itos_users, stoi_users = read.readUserDictionary()
     itos_users, stoi_users = read.readUserDictionary()
     
     
     user_index = read.keys.index("author")
     
     
     def userDataIterator():
        data = read.dataIterator(doTokenization=False, printProblems=False)
        counter = 0
        for sentence in data:
           if sentence[0] == "1":
                continue
           user_id = stoi_users.get(sentence[user_index], None)
           if user_id is not None and user_id < 1000:
             counter += 1
             if counter % 10000 == 0:
                print(counter)
             read.tokenize(sentence) # this step is the bottleneck
             tokens = sentence[1]
             for token in tokens:
                token = token.lower()
                token_id = stoi.get(token, None)
                if token_id is not None and token_id < 10000:
                    yield (token_id, user_id)
     
     import numpy as np
     import torch
     import torch.nn
     import torch.optim as optim
     
     userTrainingData = userDataIterator()
     
     unigram_probabilities = read.getUnigramProbabilities(vocab_size=10000)
     
     user_embeddings = torch.nn.Embedding(num_embeddings=1000, embedding_dim=100).cuda()
     
     optimizer = optim.SGD(user_embeddings.parameters(), lr=learning_rate)
     
     
     glove = torch.nn.Embedding(num_embeddings=10000, embedding_dim=100).cuda()
     glove.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi, offset=0)).cuda())
     
     norm = torch.norm(glove.weight.data, p=2, dim=1).unsqueeze(1)
     
     glove.weight.data = glove.weight.data.div(norm.expand_as(glove.weight.data))
     
     print("Read embeddings")
     
     
     def printClosestNeighbors():
        users = user_embeddings.weight.data[:20].view(20, 1, 1, 100) # 20 x 100
        words = glove.weight.data[:5000].view(1, 5000, 100, 1)
        scores = torch.matmul(users, words) # 20 x 10000 x 1 x 1
        scores = scores.view(20, 5000)
        best = torch.topk(scores, k=10)[1].data.cpu().numpy()
        for i in range(20):
           print("------------")
           print(itos_users[i])
           print([itos[j] for j in best[i]])
     
     
     runningAverageLoss = 1
     counterTraining = 0
     while True:
         counterTraining += 1
         optimizer.zero_grad()
         batch = [next(userTrainingData) for _ in range(batchSize)]
         tokens = [x[0] for x in batch]
         users = [x[1] for x in batch]
         negative_samples = np.random.choice(10000, size=(batchSize*number_of_negative_samples), p=unigram_probabilities)
         positive_tokens = glove(torch.LongTensor(tokens).cuda()) # (batchSize, 100)
         negative_tokens = glove(torch.LongTensor(negative_samples).cuda()) # (batchSize * negSamples, 100)
         user_embedded = user_embeddings(torch.LongTensor(users).cuda()) # (batchSize, 100)
         dotProductPositive = torch.bmm(user_embedded.unsqueeze(1), positive_tokens.unsqueeze(2)).unsqueeze(1) # (batchSize, 1, 1)
     
         first =user_embedded.unsqueeze(1).unsqueeze(2)
         second = negative_tokens.view(batchSize, number_of_negative_samples, 100, 1)
         dotProductNegative = torch.matmul(first, second)
         dotProductNegative = dotProductNegative.view(batchSize, number_of_negative_samples, 1)
     
         loss = torch.nn.ReLU()(1-dotProductPositive+dotProductNegative)
         meanLoss = loss.mean()
         meanLoss.backward()
         optimizer.step()
     
         runningAverageLoss = 0.999 * runningAverageLoss + (1-0.999) * meanLoss.data.cpu().numpy()
         if counterTraining % 1000 == 0:
            print(runningAverageLoss)
            if counterTraining % 50000 == 0:
               printClosestNeighbors()
     
               if normalizeEmbeddings:
                  norm = torch.norm(user_embeddings.weight.data, p=2, dim=1).unsqueeze(1).detach()
                  user_embeddings.weight.data = user_embeddings.weight.data.div(norm.expand_as(user_embeddings.weight.data))
     
            if counterTraining % 100000 == 0:
               learning_rate *= learning_rate_decay
               optimizer = optim.SGD(user_embeddings.parameters(), lr=learning_rate)
               print("Decaying learning rate")
               print(learning_rate)
               print("Storing embeddings in file")
               save_to_file(itos_users, user_embeddings.weight.data.cpu().numpy())
               

