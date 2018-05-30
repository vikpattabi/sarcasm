import bz2
import spacy
import zipfile

#  python -m spacy download en

from spacy.lang.en import English
nlp = English()

with open("data/raw/key.csv", "r") as inFile:
  keys = inFile.read().strip().split("\t")



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
        print("SAVING")
        words = list(wordCounts.items())
        words = sorted(words, key=lambda x:-x[1])
        itos = [x[0] for x in words]
      
        with open("data/processed/vocabulary.tsv", "w") as outFile:
            for line in itos:
               print(line, file=outFile)

def createVocabulary(vocab_size=10000):
  """
     creates the vocabulary, sorted descendingly by frequency in the large CSV file (currently only the first 10000 sentences)
  """
  with open("data/processed/vocabulary.tsv", "r") as inFile:
     itos = inFile.read().strip().split("\n")[:vocab_size]
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


def readProcessedTrainingData():
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

def loadGloveEmbeddings(stoi):
   embeddings = [None, None, None] + [None for _ in stoi]
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
          if word in stoi:
             embeddings[stoi[word]+3] = embedding
   for i in range(len(embeddings)):
       if embeddings[i] is None:
          embeddings[i] = [random.uniform(-0.01, 0.01) for _ in range(100)]
   return embeddings


# 0 is Start-of-sentence (+ padding for minibatching), 1 is end-of-sentence, 2 is out-of-vocabulary
def encode_token(token, stoi):
   if token in stoi:
      return stoi[token]+3
   else:
      return 2



def encode_sentence(sentence, stoi):
   return [0]+[encode_token(x, stoi) for x in sentence]+[1]

# Reads training data with words replaced by ints, as given in the dict passed as argument
def readTrainingAndDevData(stoi):
  training_data = []
  
  comment_index = keys.index("comment")
  parent_index = keys.index("parent_comment")
  for dataPoint in readProcessedTrainingData():
      dataPoint[comment_index] = encode_sentence(dataPoint[comment_index], stoi)
      dataPoint[parent_index] = encode_sentence(dataPoint[parent_index], stoi)
      training_data.append(dataPoint)
  
  
  held_out_data = training_data[:1000]
  training_data = training_data[1000:]

  return training_data, held_out_data  
  
