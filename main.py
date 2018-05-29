


from data import read

from architecture import model


with open("data/raw/key.csv", "r") as inFile:
  keys = inFile.read().strip().split("\t")


itos, stoi = read.createVocabulary(vocab_size=10000)
print("Read dictionary")



# 0 is Start-of-sentence (+ padding for minibatching), 1 is end-of-sentence, 2 is out-of-vocabulary
def encode_token(token):
   if token in stoi:
      return stoi[token]+3
   else:
      return 2

def encode_sentence(sentence):
   return [0]+[encode_token(x) for x in sentence]+[1]



training_data = []

comment_index = keys.index("comment")
parent_index = keys.index("parent_comment")
for dataPoint in read.readProcessedTrainingData():
    dataPoint[comment_index] = encode_sentence(dataPoint[comment_index])
    dataPoint[parent_index] = encode_sentence(dataPoint[parent_index])
    training_data.append(dataPoint)
training_data = sorted(training_data, key = lambda x:(len(x[comment_index]), len(x[parent_index])))

print("Read training data.")


import random

print("Length of training set "+str(len(training_data)))


import torch




embeddings = model.embeddings(vocab_size=10000+3, embedding_size=200).cuda()
embeddings.data = torch.FloatTensor(read.loadGloveEmbeddings(stoi)).cuda()

print("Read embeddings")

encoder = model.encoderRNN(hidden_size=200, embedding_size=200, embeddings=embeddings).cuda()
decoder = model.decoderRNN(hidden_size=200, embedding_size=200, embeddings=embeddings, vocab_size=10000+3).cuda()
lossModule = torch.nn.NLLLoss()
lossModuleNoAverage = torch.nn.NLLLoss(size_average=False)

# Visualize attention matrices

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

batchSize = 16

#training_partitions = range(len(training_data)/batchSize)
#
#shuffle(training_partitions)


def predictFromInput(input_sentence):
   input = encode_sentence(input_sentence)
   
   _, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
   generated = [torch.LongTensor([0]).cuda()]
   generated_words = []
   while True:
      input = generated[-1]
 #     print(input.size())
      output, hidden = decoder.forward(input, hidden)
#      print(output.size())
#      _, predicted = torch.max(output, 2)
      _, predicted = torch.topk(output, 2, dim=2)
#      print(predicted)
      predicted = predicted.data.cpu().view(2).numpy()
      if predicted[0] == 2:
          predicted = predicted[1]
      else:
         predicted = predicted[0]
      
      predicted_numeric = predicted
#      print(predicted_numeric)
#      print(generated_words)
      if predicted_numeric == 1 or predicted_numeric == 0 or len(generated_words) > 100:
         return generated_words
      elif predicted_numeric ==2:
        generated_words.append("OOV")
      else:
        generated_words.append(itos[predicted_numeric-3])
      generated.append(torch.LongTensor([predicted_numeric]).cuda())


held_out_data = training_data[:1000]
training_data = training_data[1000:]



devLosses = []

for epoch in range(1000):
   random.shuffle(training_data)
   
   steps = 0
   crossEntropy = 10
   for dataPoint in training_data:
      steps += 1
      
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      embeddings_optimizer.zero_grad()
   

      _, hidden = encoder.forward(torch.LongTensor(dataPoint[parent_index]).cuda(), None)
      output, _ = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]).cuda(), None)
      output = output.view(-1, 10000+3)
      target = torch.LongTensor(dataPoint[comment_index][1:]).view(-1).cuda()

      loss = lossModule(output, target)
      crossEntropy = 0.99 * crossEntropy + (1-0.99) * loss.data.cpu().numpy()
   
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()
      embeddings_optimizer.step()
   
   
      if steps % 1000 == 0:
          print((epoch,steps,crossEntropy))
          print(devLosses)
          print(predictFromInput(["This", "article", "is", "such", "BS", "."]))
          print(predictFromInput(["This", "article", "is", "awesome", "."]))
          print(predictFromInput(["Bankers", "celebrate", "the", "start", "of", "the", "Trump", "era", "."]))

   steps = 0
   totalLoss = 0
   numberOfWords = 0
   for dataPoint in held_out_data:
      steps += 1
   
      numberOfWords += len(dataPoint[comment_index])-1

      _, hidden = encoder.forward(torch.LongTensor(dataPoint[parent_index]).cuda(), None)
      output, _ = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]).cuda(), None)
      output = output.view(-1, 10000+3)
      target = torch.LongTensor(dataPoint[comment_index][1:]).view(-1).cuda()

      loss = lossModuleNoAverage(output, target)
      crossEntropy = 0.99 * crossEntropy + (1-0.99) * loss.data.cpu().numpy()
      totalLoss +=  loss.data.cpu().numpy()
   devLosses.append(totalLoss/numberOfWords)
   print(devLosses)
   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
       print("Overfitting, stop")



   
