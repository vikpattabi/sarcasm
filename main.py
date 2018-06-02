from data import read
from user_info import train_embeddings
itos, stoi = read.createVocabulary(vocab_size=10000) # restrict to 10000 most frequent words
print("Read dictionary")

#itos_subreddits, stoi_subreddits = read.readUserDictionary()
itos_subreddits, stoi_subreddits = read.readSubredditDictionary()

trainer = train_embeddings.subredditEmbeddings(200, itos, stoi, stoi_subreddits)
# should have a subreddit embedding matrix

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
#           print(token)
           token_id = stoi.get(token, None)
 #          print(token_id)
  #         print(stoi)
           if token_id is not None and token_id < 10000:
               yield (token_id, subreddit_id)

import numpy as np
import torch
import torch.nn
import torch.optim as optim

subredditTrainingData = subredditDataIterator()

unigram_probabilities = read.getUnigramProbabilities(vocab_size=10000)

#model = embeddings_model.embeddingModel(self.embedding_size, self.vocabulary, 15)

subreddit_embeddings = torch.nn.Embedding(num_embeddings=1000, embedding_dim=100).cuda()


optim = optim.SGD(subreddit_embeddings.parameters(), lr=0.001)


glove = torch.nn.Embedding(num_embeddings=10000, embedding_dim=100).cuda()
glove.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi, offset=0)).cuda())
print("Read embeddings")


def printClosestNeighbors():
   subreddits = subreddit_embeddings.weight.data[:20].view(20, 1, 1, 100) # 20 x 100
   words = glove.weight.data.view(1, 10000, 100, 1)
   scores = torch.matmul(subreddits, words) # 20 x 10000 x 1 x 1
   scores = scores.view(20, 10000)
   best = torch.topk(scores, k=10)[1].data.cpu().numpy()
   for i in range(20):
      print("------------")
      print(itos_subreddits[i])
      print([itos[j] for j in best[i]])
#   quit()

batchSize = 10
number_of_negative_samples = 15

runningAverageLoss = 1
counterTraining = 0
while True:
    counterTraining += 1
    optim.zero_grad()
    batch = [next(subredditTrainingData) for _ in range(batchSize)]
    tokens = [x[0] for x in batch]
    subreddits = [x[1] for x in batch]
    negative_samples = np.random.choice(10000, size=(batchSize*number_of_negative_samples), p=unigram_probabilities)
    positive_tokens = glove(torch.LongTensor(tokens).cuda()) # (batchSize, 100)
    negative_tokens = glove(torch.LongTensor(negative_samples).cuda()) # (batchSize * negSamples, 100)
    subreddit_embedded = subreddit_embeddings(torch.LongTensor(subreddits).cuda()) # (batchSize, 100)
    dotProductPositive = torch.bmm(subreddit_embedded.unsqueeze(1), positive_tokens.unsqueeze(2)).unsqueeze(1) # (batchSize, 1, 1)

    first =subreddit_embedded.unsqueeze(1).unsqueeze(2)
#    print(first.size())
    second = negative_tokens.view(batchSize, number_of_negative_samples, 100, 1)
 #   print(second.size())
    dotProductNegative = torch.matmul(first, second)
  #  print(dotProductNegative.size())
    dotProductNegative = dotProductNegative.view(batchSize, number_of_negative_samples, 1)

    loss = torch.nn.ReLU()(1-dotProductPositive+dotProductNegative)
    meanLoss = loss.mean()
    meanLoss.backward()
    optim.step()

    runningAverageLoss = 0.999 * runningAverageLoss + (1-0.999) * meanLoss.data.cpu().numpy()
    if counterTraining % 1000 == 0:
       print(runningAverageLoss)
       if counterTraining % 50000 == 0:
          printClosestNeighbors()

#    print(loss)
#
#    print(negative_samples)

#     print(tokens)

#     print(subreddit_id)
#   print(sentence)

quit()


def extractDataForTopUsers():
    itos_users, stoi_users = read.readUserDictionary()
    
    author_index = read.keys.index("author")
    
    data = read.dataIterator(doTokenization=False, printProblems=False)
    counter = 0

    with open("data/processed/dataForUsers.tsv", "w") as outFile:
       for sentence in data:
#          if sentence[0] == "1":
#             continue
          user = sentence[author_index]
          user_id = stoi_users.get(user, None)
#          print(user)
          if user_id is not None:
            print(user_id)
          if user_id is not None and user_id < 1000:
             counter += 1
             if counter % 10 == 0:
                print(counter)
             outFile.write(("\t".join(sentence))+"\n")
 
extractDataForTopUsers()

quit()


from data import read

from architecture import model

# Read the word list
# itos is a list of words
# stoi (dict) maps words (strings) to their position in itos (int)
itos, stoi = read.createVocabulary(vocab_size=10000) # restrict to 10000 most frequent words
print("Read dictionary")

# Reading the data that we have extracted to far
training_data, held_out_data = read.readTrainingAndDevData(stoi)
print("Read training data.")
print("Length of training set "+str(len(training_data)))

number_of_subreddits = 1000
subreddits = {}
subreddit_index = read.keys.index("subreddit")
for dataPoint in training_data + held_out_data:
  subreddits[dataPoint[subreddit_index]] = subreddits.get(dataPoint[subreddit_index], 0) + 1
#print(subreddits)
subreddits = sorted(list(subreddits.items()), key=lambda x:x[1], reverse=True)[:number_of_subreddits]
#print("\n".join(["\t".join([x[0],str(x[1])]) for x in subreddits]))
#print("SUBREDDITS2")
#print(subreddits)
itos_subreddits = [x[0] for x in subreddits]
#print(itos_subreddits)
stoi_subreddits = dict([(itos_subreddits[i], i) for i in range(len(itos_subreddits))])
#print("\n".join(itos_subreddits))

#print("AUTHORS")
number_of_authors = 10000
authors = {}
author_index = read.keys.index("author")
for dataPoint in training_data + held_out_data:
  authors[dataPoint[author_index]] = authors.get(dataPoint[author_index], 0) + 1
#print(authors)
authors = sorted(list(authors.items()), key=lambda x:x[1], reverse=True)[:number_of_authors]
#print("\n".join(["\t".join([x[0], str(x[1])]) for x in authors]))
#print("AUTHORS2")
itos_authors = [x[0] for x in authors]
#print(itos_authors)
stoi_authors = dict([(itos_authors[i], i) for i in range(len(itos_authors))])
#print("\n".join(itos_authors))





from architecture import training
import random
import torch


useGlove = True

useSubredditEmbeddings = True

embeddings = model.embeddings(vocab_size=10000+3, embedding_size=100).cuda()
if useGlove:
   embeddings.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi)).cuda())

print("Read embeddings")

if useSubredditEmbeddings:
  subreddit_embeddings = model.embeddings(vocab_size=number_of_subreddits+1, embedding_size=100).cuda()
else:
  subreddit_embeddings = None


useAttention = False #True

encoder = model.encoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings).cuda()
if useAttention:
  decoder = model.attentionDecoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings, vocab_size=10000+3).cuda()
else:
  decoder = model.decoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings, vocab_size=10000+3).cuda()




training.run_training_loop(training_data, held_out_data, encoder, decoder, embeddings, batchSize=128, learning_rate=0.001, optimizer="Adam", useAttention=useAttention, stoi=stoi, itos=itos, subreddit_embeddings=subreddit_embeddings, stoi_subreddits=stoi_subreddits, itos_subreddits=itos_subreddits)


