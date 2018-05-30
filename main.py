


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


from architecture import training
import random
import torch


useGlove = True

embeddings = model.embeddings(vocab_size=10000+3, embedding_size=100).cuda()
if useGlove:
   embeddings.weight.data.copy_(torch.FloatTensor(read.loadGloveEmbeddings(stoi)).cuda())

print("Read embeddings")


useAttention = True

encoder = model.encoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings).cuda()
if useAttention:
  decoder = model.attentionDecoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings, vocab_size=10000+3).cuda()
else:
  decoder = model.decoderRNN(hidden_size=200, embedding_size=100, embeddings=embeddings, vocab_size=10000+3).cuda()




training.run_training_loop(training_data, held_out_data, encoder, decoder, embeddings, batchSize=32, learning_rate=0.001, optimizer="Adam", useAttention=useAttention, stoi=stoi, itos=itos)


