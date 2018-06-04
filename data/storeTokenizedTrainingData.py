from data import read


training_data = []

comment_index = read.keys.index("comment")
parent_index = read.keys.index("parent_comment")
subreddit_index = read.keys.index("subreddit")
author_index = read.keys.index("author")

counter = 0
with open("data/processed/tokenized.txt", "w") as outFile:
  for dataPoint in read.readProcessedTrainingData():
    outFile.write((" ".join([dataPoint[subreddit_index], dataPoint[author_index]] + dataPoint[comment_index] + ["__PARENT__"] + dataPoint[parent_index])) + "\n")
    counter += 1
    if counter % 10000 == 0:
        print(counter) 


