# Extract all sarcastic comments from the big csv


import time


import read

started_at = time.time()
counter = 0
start_at_index = 0

found_sarcastic = 0

with open("data/processed/all-sarcastic.tsv", "w") as outFile:
   for comment in read.dataIterator(doTokenization=False, printProblems=False):
      if comment[0] == "1":
          outFile.write(("\t".join(comment))+"\n")
          found_sarcastic += 1
      if counter < start_at_index:
          continue
      counter += 1
      if counter % 1e5 == 0:
         speed = float(found_sarcastic)/(time.time()-started_at)
         print((counter, speed, float(found_sarcastic)/1e6), ((1e6-found_sarcastic)/speed)/3600 if speed > 0 else "INFINITY")
  
