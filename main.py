


from data import read

from architecture import model

# extract the context comments, and the sarcastic response (not the nonsarcastic one)

#[['4o8qvz'], ['d4b1i6p', 'd4b4mf8'], ['0', '1']]

comments_code_to_text = {}

expected_sentences = 0

trainingData = read.readTrainingData()
trainingData_as_list = []
index = 0
for dataPoint in trainingData:
   for context in dataPoint[0]:
        comments_code_to_text[context] = index
        expected_sentences += 1
   for i, comment in enumerate(dataPoint[1]):
      if dataPoint[2][i] == "1":
         comments_code_to_text[comment] = index
         expected_sentences += 1
   index += 1
   if index % 10000 == 0:
       print(index)
   trainingData_as_list.append(dataPoint)
   
#   print(comments_code_to_text)
#   for i in range(len(dataPoint[1])):
#      if dataPoint[1][i] in idToComments:
#         print(1)
#      else:
#         print(2)


# ['0', ['"', 'Now', 'that', 'I', 'think', 'about', 'it', 'that', 'way', '...', 'FFFFFFFUUUUUUUU', '"'], 'relic2279', 'AskReddit', '3', '3', '0', '2009-01', '1233384488', [], 'c07ds4k', '7tqtr']


#print(trainingData_as_list)

with open("data/raw/key.csv", "r") as inFile:
  keys = inFile.read().strip().split("\t")


id_index = keys.index("id")
link_id_index = keys.index("link_id")

sentences_found = 0

import time

started_at = time.time()
counter = 0
start_at_index = 0 #1e10
for comment in read.dataIterator(doTokenization=False, printProblems=False):
   if counter < start_at_index:
       continue
   counter += 1
   if counter % 20000 == 0:
      speed = float(sentences_found)/(time.time()-started_at)
      print((counter, float(sentences_found)/expected_sentences), ((expected_sentences-sentences_found)/speed)/3600 )
   if comment[id_index] in comments_code_to_text:
 #       print("ID")
#        print(comment)
        sentences_found += 1
        dataPoint = trainingData_as_list[comments_code_to_text[comment[id_index]]]
        id_of_sentence = comment[id_index]
        if id_of_sentence in dataPoint[0]:
            dataPoint[0][dataPoint[0].index(id_of_sentence)] = comment
        elif id_of_sentence in dataPoint[1]:
            dataPoint[1][dataPoint[1].index(id_of_sentence)] = comment
        else:
            assert False
   if counter % 1e6 == 0:
        print("SAVING")
        with open("data/processed/partial"+str(start_at_index)+"-"+str(counter), "w") as outFile:
          for dataPoint in trainingData_as_list:
             for comment in dataPoint[0]:
                if type(comment) is list:
                   outFile.write("###CONTEXT###")
                   outFile.write("\t".join(comment))
             for response, annotation in zip(dataPoint[1], dataPoint[2]):
                if type(response) is list:
                   outFile.write("###SARCASTIC_RESPONSE###" if annotation == "1" else "###NON_SARCASTIC_RESPONSE###")
                   outFile.write("\t".join(response))
             outFile.write("\n") 
               
           
#   elif  comment[link_id_index] in comments_code_to_text:
#        print("LINK_ID")
#        print(comment)

quit()


#read.createAndStoreVocabulary()
#quit()

# Creates a vocabulary sorted by frequency, where
#  itos (list) maps ints to words (strings), with words ordered by frequency in descending order
#  stoi (dict) maps words to ints
itos, stoi = read.createVocabulary()
#print(itos)

# reads comments from the large CSV file
comments = read.readComments()


idToComments = {}
idIndex = keys.index("id")
for comment in comments:
   idToComments[comment[idIndex]] = comment
print(idToComments)

trainingData = read.readTrainingData()
for dataPoint in trainingData:
   print(dataPoint)
   for i in range(len(dataPoint[1])):
      if dataPoint[1][i] in idToComments:
         print(1)
#      else:
#         print(2)


# ['0', ['Yeah', '.', 'Trolls', 'are', 'funny', ',', 'though', '.'], 'NitsujTPU', 'funny', '1', '1', '0', '2009-01', '1233084256', '*sniff sniff*  smells like, TROLL!', 'c07b31t', '7srzd']






#import torch

