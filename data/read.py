import bz2
import spacy
import zipfile


import random

#  python -m spacy download en

from spacy.lang.en import English
nlp = English()

keys = "label	comment	author	subreddit	score	ups	downs	date	created_utc	parent_comment	id	link_id".split("\t")


def tokenize(line, response=True, parent=False): # in-place operation
         if response:
            tokens = nlp(line[1])
            tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
            line[1] = tokens
         if parent:
            tokens = nlp(line[9])
            tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
            line[9] = tokens


def dataIterator(doTokenization=True, printProblems=True):
   """
      reads the large CSV file
   """
   with bz2.BZ2File("data/raw/sarc.csv.bz2", "r") as inFile:
      while True:
         line = []
         while len(line) < 12:
            line2 = next(inFile)
            line2 = line2.decode("utf-8").strip()
            line2 = line2.split("\t")
            if len(line) > 0:
              line[-1] += line2[0]
              line = line+line2[1:]
            else:
              line = line2
         if len(line) > 12:
            if printProblems:
               print(line)
            continue

#         while len(line) > 12:
#            line[1] = line[1]+line[2]
#            del line[1]
#         print(line)
         if len(line) < 2:
            continue
         if doTokenization:
            tokens = nlp(line[1])
            tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
            line[1] = tokens
   
            tokens = nlp(line[9])
            tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
            line[9] = tokens



#         print(tokens)
         assert len(line) == 12, line
#         print(len(line))
#         print(line)
         yield line


def readComments():
  """
     reads the large CSV file into a list (currently only the first 10000 sentences)
  """
  iterator = dataIterator()
  comments = []
  for i in range(10000):
     comments.append(next(iterator))
  return comments


def createAndStoreVocabulary(save_period):
  wordCounts = {}
  iterator = dataIterator()
  i = 0
  while True:
     i += 1
     if i % 5000 == 0:
        print(i)
     for word in next(iterator)[1]:
        wordCounts[word] = wordCounts.get(word,0)+1
     if i % save_period == 0:
        print("SAVING "+"data/processed/vocabulary-with-counts.tsv")
        words = list(wordCounts.items())
        words = sorted(words, key=lambda x:-x[1])
#        itos = [x[0] for x in words]
      
        with open("data/processed/vocabulary-with-counts.tsv", "w") as outFile:
            for line in words:
               print(line[0]+"\t"+str(line[1]), file=outFile)

def createVocabulary(vocab_size=10000):
  """
     creates the vocabulary, sorted descendingly by frequency in the large CSV file (currently only the first 10000 sentences)
  """
  with open("data/processed/vocabulary-with-counts-lower.tsv", "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:vocab_size]]
  stoi = dict([(itos[i], i) for i in range(len(itos))])
  return itos , stoi

def readTrainingData():
   """
      reads the training data
      Each line is split by |, and within that by whitespace.
   """
   with bz2.BZ2File("data/main/train-unbalanced.csv.bz2", "r") as inFile:
      for line in inFile:
         line = [x.split(" ") for x in line.decode("utf-8").strip().split("|")]
#         print(line)
         assert len(line) == 3
         assert len(line[2]) == len(line[1]), line
         yield line



def readProcessedTrainingData(fileName="all-sarcastic-200K.tsv"):
   with open("data/processed/"+fileName, "r") as data:
      for response in data:
                response = response.split("\t")
                assert len(response) == 12, response
                if len(response[9]) < 2:
                    continue
                tokens = nlp(response[1])
                tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
                response[1] = tokens
       
                tokens = nlp(response[9])
                tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
                response[9] = tokens
                
 

                yield response


def readProcessedTrainingDataOld():
   with open("data/processed/partial0-220000000", "r") as data:
      for line in data:
          if len(line) > 1:
             line = line.strip().split("###SARCASTIC_RESPONSE###")
             for response in line:
                if len(response) == 0:
                    continue
                response = response.split("\t")
                assert len(response) == 12, response
                tokens = nlp(response[1])
                tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
                response[1] = tokens
       
                tokens = nlp(response[9])
                tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
                response[9] = tokens
 

                yield response

import random

def loadGloveEmbeddings(stoi, offset=3):
   embeddings = [None for _ in range(offset)] + [None for _ in stoi]
   zipFile = zipfile.ZipFile("data/embeddings/glove.6B.zip", "r")
   counter = 0
   with zipFile.open("glove.6B.100d.txt", "r") as inFile:
      for line in inFile:
          counter += 1
          if counter % 50000 == 0:
              break
              print(counter)
          line = line.decode("utf8").split(" ")
          word = line[0]
          embedding = list(map(float,line[1:]))
          entry = stoi.get(word, None)
          if entry is not None:
             embeddings[entry+offset] = embedding
   for i in range(len(embeddings)):
       if embeddings[i] is None:
          embeddings[i] = [random.uniform(-0.01, 0.01) for _ in range(100)]
   return embeddings


# 0 is Start-of-sentence (+ padding for minibatching), 1 is end-of-sentence, 2 is out-of-vocabulary
def encode_token(token, stoi):
   token = token.lower()
   if token in stoi:
      return stoi[token]+3
   else:
      return 2

maximumAllowedLength = 40

def encode_sentence(sentence, stoi):
   sentence = sentence[:maximumAllowedLength]
   return [0]+[encode_token(x, stoi) for x in sentence]+[1]

# Reads training data with words replaced by ints, as given in the dict passed as argument
def readTrainingAndDevData(stoi):
  training_data = []
  
  comment_index = keys.index("comment")
  parent_index = keys.index("parent_comment")
  counter = 0
  for dataPoint in readProcessedTrainingData():
      counter += 1
      if counter % 10000 == 0:
          print(counter) 
      dataPoint[comment_index] = encode_sentence(dataPoint[comment_index], stoi)
      dataPoint[parent_index] = encode_sentence(dataPoint[parent_index], stoi)
      training_data.append(dataPoint)
  
  random.Random(5).shuffle(training_data)
  held_out_data = training_data[:1000]
  training_data = training_data[1000:]

  return training_data, held_out_data  



def readTrainingAndDevDataTokenized(stoi, bound=None):
   training_data = []
   
   comment_index = keys.index("comment")
   parent_index = keys.index("parent_comment")
   subreddit_index = keys.index("subreddit")
   author_index = keys.index("author")
   
   counter = 0
   with open("data/processed/tokenized-all-shuffled-train.txt", "r") as outFile:
     for line in outFile:
       counter += 1
       if counter % 10000 == 0:
           print(counter) 
 
       line = line.strip().split(" ")
       dataPoint = [None for _ in keys]
       dataPoint[subreddit_index] = line[0]
       dataPoint[author_index] = line[1]
       parentStart = line.index("__PARENT__")
       dataPoint[comment_index] = encode_sentence(line[2:parentStart], stoi)
       dataPoint[parent_index] = encode_sentence(line[parentStart+1:], stoi)

       training_data.append(dataPoint)

   held_out_data = []
   with open("data/processed/tokenized-all-shuffled-dev.txt", "r") as outFile:
     for line in outFile:
       counter += 1
       if counter % 10000 == 0:
           print(counter) 
 
       line = line.strip().split(" ")
       dataPoint = [None for _ in keys]
       dataPoint[subreddit_index] = line[0]
       dataPoint[author_index] = line[1]
       parentStart = line.index("__PARENT__")
       dataPoint[comment_index] = encode_sentence(line[2:parentStart], stoi)
       dataPoint[parent_index] = encode_sentence(line[parentStart+1:], stoi)

       held_out_data.append(dataPoint)
   assert len(training_data) > 0
   assert len(held_out_data) > 0

 
   return training_data, held_out_data  
 


def readTrainingAndDevDataTokenizedOld(stoi):
   training_data = []
   
   comment_index = keys.index("comment")
   parent_index = keys.index("parent_comment")
   subreddit_index = keys.index("subreddit")
   author_index = keys.index("author")
   
   counter = 0
   with open("data/processed/tokenized.txt", "r") as outFile:
     for line in outFile:
       counter += 1
       if counter % 10000 == 0:
           print(counter) 
 
       line = line.strip().split(" ")
       dataPoint = [None for _ in keys]
       dataPoint[subreddit_index] = line[0]
       dataPoint[author_index] = line[1]
       parentStart = line.index("__PARENT__")
       dataPoint[comment_index] = encode_sentence(line[2:parentStart], stoi)
       dataPoint[parent_index] = encode_sentence(line[parentStart+1:], stoi)

       training_data.append(dataPoint)
   random.Random(5).shuffle(training_data)
   held_out_data = training_data[:1000]
   training_data = training_data[1000:]
 
   return training_data, held_out_data  
 



def readUserDictionary():
   with open("data/processed/users-counts.tsv", "r") as inFile:
      users = inFile.read().strip().split("\n")
   itos_users = [x.split("\t")[0] for x in users]
   stoi_users = dict([(itos_users[i], i) for i in range(len(itos_users))])
   return itos_users, stoi_users

def readSubredditDictionary():
   with open("data/processed/subreddit-counts.tsv", "r") as inFile:
      subreddits = inFile.read().strip().split("\n")
   itos_subreddits = [x.split("\t")[0] for x in subreddits]
   stoi_subreddits = dict([(itos_subreddits[i], i) for i in range(len(itos_subreddits))])
   return itos_subreddits, stoi_subreddits

import numpy as np

def getUnigramProbabilities(vocab_size=10000):
   with open("data/processed/vocabulary-with-counts-lower.tsv", "r") as inFile:
      counts = np.asarray([int(x.split("\t")[1]) for x in inFile.read().strip().split("\n")[:vocab_size]])
      total = sum(counts)
      return counts/total
      


def loadSubredditEmbeddings(stoi, offset=1):
   embeddings = [None for _ in range(offset)] + [None for _ in stoi]
   counter = 0
   with open("data/embeddings/subreddit_embeddings.txt", "r") as inFile:
      for line in inFile:
          counter += 1
          if counter % 50000 == 0:
              break
              print(counter)
          line = line.split(" ")
          word = line[0]
          embedding = list(map(float,line[1:]))
          entry = stoi.get(word, None)
          if entry is not None:
             embeddings[entry+offset] = embedding
   for i in range(len(embeddings)):
       if embeddings[i] is None:
          embeddings[i] = [random.uniform(-0.01, 0.01) for _ in range(100)]
   return embeddings




