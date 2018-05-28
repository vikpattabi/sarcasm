


import read


# extract the context comments, and the sarcastic response (not the nonsarcastic one)

comments_code_to_text = {}

expected_sentences = 0
expected_contexts = 0
expected_comments = 0
trainingData = read.readTrainingData()
trainingData_as_list = []
index = 0
for dataPoint in trainingData:
   #for context in dataPoint[0]:
   #     comments_code_to_text[context] = index
   #     expected_sentences += 1
   #     expected_contexts += 1
   for i, comment in enumerate(dataPoint[1]):
      if dataPoint[2][i] == "1":
         comments_code_to_text[comment] = index
         expected_sentences += 1
#         expected_comments += 1
   index += 1
   if index % 10000 == 0:
       print(index)
   trainingData_as_list.append(dataPoint)
   

with open("data/raw/key.csv", "r") as inFile:
  keys = inFile.read().strip().split("\t")


id_index = keys.index("id")
link_id_index = keys.index("link_id")

sentences_found = 0
context_found = 0
comments_found = 0
import time

started_at = time.time()
counter = 0
start_at_index = 0
for comment in read.dataIterator(doTokenization=False, printProblems=False):
   if counter < start_at_index:
       continue
   counter += 1
   if counter % 1e5 == 0:
      speed = float(sentences_found)/(time.time()-started_at)
      print((counter, expected_sentences, float(sentences_found)/expected_sentences), ((expected_sentences-start_at_index-sentences_found)/speed)/3600 if speed > 0 else "INFINITY")
   if comment[id_index] in comments_code_to_text:
 #       print("ID")
#        print(comment)
        sentences_found += 1
        dataPoint = trainingData_as_list[comments_code_to_text[comment[id_index]]]
        id_of_sentence = comment[id_index]

   #     if id_of_sentence in dataPoint[0]:
   #         dataPoint[0][dataPoint[0].index(id_of_sentence)] = comment
        if id_of_sentence in dataPoint[1]:
            dataPoint[1][dataPoint[1].index(id_of_sentence)] = comment
        else:
            assert False
   #     print(dataPoint)
#   elif comment[link_id_index] in comments_code_to_text:
# #       print("ID")
##        print(comment)
#        sentences_found += 1
#        dataPoint = trainingData_as_list[comments_code_to_text[comment[link_id_index]]]
#        link_id_of_sentence = comment[link_id_index]
#
#        if link_id_of_sentence in dataPoint[0]:
#            dataPoint[0][dataPoint[0].index(link_id_of_sentence)] = comment
#        elif link_id_of_sentence in dataPoint[1]:
#            dataPoint[1][dataPoint[1].index(link_id_of_sentence)] = comment
#        else:
#            assert False
#        print(dataPoint)

   if counter % 1e7 == 0:
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
               
           

