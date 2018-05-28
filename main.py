


from data import read

from architecture import model


with open("data/raw/key.csv", "r") as inFile:
  keys = inFile.read().strip().split("\t")

itos, stoi = read.createVocabulary(vocab_size=10000)
print(itos)



# 0 is Start-of-sentence (+ padding for minibatching), 1 is end-of-sentence, 2 is out-of-vocabulary
def encode_token(token):
   if token in stoi:
      return stoi[token]+3
   else:
      return 2

training_data = []

comment_index = keys.index("comment")
parent_index = keys.index("parent_comment")
for dataPoint in read.readProcessedTrainingData():
    dataPoint[comment_index] = [0]+[encode_token(x) for x in dataPoint[comment_index]]+[1]
    dataPoint[parent_index] = [0]+[encode_token(x) for x in dataPoint[parent_index]]+[1]
    training_data.append(dataPoint)
print(training_data) 

#for datapoint in training_data#
#    print(datapoint)


import torch

embeddings = model.embeddings(vocab_size=10000+3, embedding_size=200)
encoder = model.encoderRNN(hidden_size=200, embedding_size=200, embeddings=embeddings)
decoder = model.decoderRNN(hidden_size=200, embedding_size=200, embeddings=embeddings, vocab_size=10000+3)
lossModule = torch.nn.NLLLoss()


learning_rate = 0.001



encoder_optimizer = None
decoder_optimizer = None
embeddings_optimizer = None

optimizer = "Adam"


if optimizer == 'Adam':
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
    embeddings_optimizer = torch.optim.Adam(embeddings.parameters(), lr = learning_rate)
else:
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = learning_rate)
    embeddings_optimizer = torch.optim.SGD(embeddings.parameters(), lr = learning_rate)



for dataPoint in training_data:
   encoder_optimizer.zero_grad()
   decoder_optimizer.zero_grad()
   embeddings_optimizer.zero_grad()



#   print(torch.LongTensor(dataPoint[comment_index]))
   _, hidden = encoder.forward(torch.LongTensor(dataPoint[parent_index]), None)
   output, _ = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]), None)
   output = output.view(-1, 10000+3)
   target = torch.LongTensor(dataPoint[comment_index][1:]).view(-1)
#   print(output)
#   print(target)
#   print(output.size())
#   print(target.size())
   loss = lossModule(output, target)
   print(loss)

   loss.backward()
   encoder_optimizer.step()
   decoder_optimizer.step()
   embeddings_optimizer.step()




