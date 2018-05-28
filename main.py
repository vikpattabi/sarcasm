


from data import read

from architecture import model


# Creates a vocabulary sorted by frequency, where
#  itos (list) maps ints to words (strings), with words ordered by frequency in descending order
#  stoi (dict) maps words to ints
itos, stoi = read.createVocabulary()
#print(itos)

with open("data/raw/key.csv", "r") as inFile:
  keys = inFile.read().strip().split("\t")
# reads comments from the large CSV file
comments = read.readComments()


idToComments = {}
idIndex = keys.index("id")
for comment in comments:
   idToComments[comment[idIndex]] = comment
print(idToComments)

trainingData = read.readTrainingData()
for dataPoint in trainingData:
#   print(dataPoint)
   for i in range(len(dataPoint[1])):
      if dataPoint[1][i] in idToComments:
         print(1)
#      else:
#         print(2)


# ['0', ['Yeah', '.', 'Trolls', 'are', 'funny', ',', 'though', '.'], 'NitsujTPU', 'funny', '1', '1', '0', '2009-01', '1233084256', '*sniff sniff*  smells like, TROLL!', 'c07b31t', '7srzd']






import torch

