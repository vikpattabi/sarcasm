from user_info import train_user_embeddings
from user_info import train_embeddings

from data import read


import argparse

from user_info import visualize_embeddings 


visualize_embeddings.trainSubredditEmbeddings(loadFromFile=True)
quit()

parser = argparse.ArgumentParser()
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str)
parser.add_argument("--attention", action='store_true')
parser.add_argument("--freeze-subreddit-embeddings", dest="freeze_subreddit_embeddings", action='store_true')
args=parser.parse_args()
print(args)

# (1) for training subreddit embeddings:
#train_embeddings.trainSubredditEmbeddings()

# (2) for training user embeddings:
#train_user_embeddings.trainUserEmbeddings()


# (3) The following code is for training the encoder-decoder model. Currently without user/subreddit embeddings

from architecture import model

# Read the word list
# itos is a list of words
# stoi (dict) maps words (strings) to their position in itos (int)
itos, stoi = read.createVocabulary(vocab_size=10000) # restrict to 10000 most frequent words
print("Read dictionary")

# Reading the data that we have extracted to far
print("Now reading training data")
training_data, held_out_data = read.readTrainingAndDevDataTokenized(stoi)
print("Read training data.")
print("Length of training set "+str(len(training_data)))

from architecture import training
import random
import torch

print("Now reading Glove embeddings")
useGlove = True
embeddings = model.embeddings(vocab_size=10000+3, embedding_size=100).cuda()
if useGlove:
   embeddings.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi)).cuda())
print("Read Glove embeddings")

useAttention = args.attention


itos_subreddits, stoi_subreddits = read.readSubredditDictionary()
subreddit_embeddings = model.embeddings(vocab_size=1000+1, embedding_size=100).cuda()
subreddit_embeddings.weight.data.copy_(torch.FloatTensor(read.loadSubredditEmbeddings(stoi_subreddits, offset=1)).cuda())
print("Read subreddit embeddings")

encoder = model.encoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings).cuda()
if useAttention:
  decoder = model.attentionDecoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings, vocab_size=10000+3, subreddit_embeddings=subreddit_embeddings).cuda()
else:
  decoder = model.decoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings, vocab_size=10000+3, subreddit_embeddings=subreddit_embeddings).cuda()



training.run_training_loop(training_data, held_out_data, encoder, decoder, embeddings, batchSize=128, learning_rate=0.001, optimizer="Adam", useAttention=useAttention, stoi=stoi, itos=itos, subreddit_embeddings=subreddit_embeddings, itos_subreddits=itos_subreddits, stoi_subreddits=stoi_subreddits, args=args)
# , subreddit_embeddings=subreddit_embeddings, stoi_subreddits=stoi_subreddits, itos_subreddits=itos_subreddits



###############################################################################################


def collectSubreddits():
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

def collectAuthors():
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
    
 
