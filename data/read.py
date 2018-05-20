import bz2

import spacy


#  python -m spacy download en


from spacy.lang.en import English
nlp = English()


def dataIterator():
   """
      reads the large CSV file
   """
   with bz2.BZ2File("data/raw/sarc_09-12.csv.bz2", "r") as inFile:
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
            print(line)
            continue

#         while len(line) > 12:
#            line[1] = line[1]+line[2]
#            del line[1]
#         print(line)
         if len(line) < 2:
            continue
         tokens = nlp(line[1])
         tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
         line[1] = tokens
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


def createVocabulary():
  """
     creates the vocabulary, sorted descendingly by frequency in the large CSV file (currently only the first 10000 sentences)
  """
  wordCounts = {}
  iterator = dataIterator()
  for i in range(10000):
     for word in next(iterator)[1]:
        wordCounts[word] = wordCounts.get(word,0)+1
  words = list(wordCounts.items())     
  words = sorted(words, key=lambda x:-x[1])
  itos = [x[0] for x in words]
  stoi = dict([(itos[i], i) for i in range(len(itos))])
  return itos, stoi

def readTrainingData():
   """
      reads the training data
      Each line is split by |, and within that by whitespace.
   """
   with bz2.BZ2File("data/main/train-balanced.csv.bz2", "r") as inFile:
      for line in inFile:
         line = [x.split(" ") for x in line.decode("utf-8").strip().split("|")]
         print(line)
         assert len(line) == 3
         assert len(line[2]) == 2
         yield line


