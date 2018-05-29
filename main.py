


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


held_out_data = training_data[:1000]
training_data = training_data[1000:]



print("Read training data.")


import random

print("Length of training set "+str(len(training_data)))


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
lossModule = torch.nn.NLLLoss(ignore_index=0)
lossModuleNoAverage = torch.nn.NLLLoss(size_average=False, ignore_index=0)

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


def predictFromInput(input_sentence):
   input = encode_sentence(input_sentence)
   
   encoder_outputs, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
   generated = [torch.LongTensor([0]).cuda()]
   generated_words = []
   while True:
      input = generated[-1]
      if not useAttention:
         output, hidden = decoder.forward(input, hidden)
      else:
         output, hidden, attention = decoder.forward(input.view(1,1), hidden, encoder_outputs=encoder_outputs)
         print(attention[0].view(-1).data.cpu().numpy()[:])
      _, predicted = torch.topk(output, 2, dim=2)
      predicted = predicted.data.cpu().view(2).numpy()
      if predicted[0] == 2:
          predicted = predicted[1]
      else:
         predicted = predicted[0]
      
      predicted_numeric = predicted
      if predicted_numeric == 1 or predicted_numeric == 0 or len(generated_words) > 100:
         return " ".join(generated_words)
      elif predicted_numeric ==2:
        generated_words.append("OOV")
      else:
        generated_words.append(itos[predicted_numeric-3])
      generated.append(torch.LongTensor([predicted_numeric]).cuda())


training_data = sorted(training_data, key = lambda x:(len(x[comment_index]), len(x[parent_index])))
held_out_data = sorted(held_out_data, key = lambda x:(len(x[comment_index]), len(x[parent_index])))



devLosses = []

batchSize = 16
training_partitions = list(range(int(len(training_data)/batchSize)))

def collectAndPadInput(current, index):
      maxLength = max([len(x[index]) for x in current])
      context_sentence = []
      for i in range(maxLength):
         context_sentence.append([x[index][i] if i < len(x[index]) else 0 for x in current])
      return context_sentence



for epoch in range(1000):

   random.shuffle(training_partitions)

  
   steps = 0
   crossEntropy = 10
   for partition in training_partitions:
      current = training_data[partition*batchSize:(partition+1)*batchSize] # reads a minibatch of length batchSize
      steps += 1
      
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      embeddings_optimizer.zero_grad()
   

      context_sentence = collectAndPadInput(current, parent_index)
      response_sentence = collectAndPadInput(current, comment_index)


      encoder_outputs, hidden = encoder.forward(torch.LongTensor(context_sentence).cuda(), None)

      response = torch.LongTensor(response_sentence).cuda()
     

      if not useAttention:
         output, _ = decoder.forward(response[:-1], hidden)
      else:
         output, _, attentions = decoder.forward(response[:-1], hidden, encoder_outputs=encoder_outputs)

      loss = lossModule(output.view(-1, 10000+3), response[1:].view(-1))


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

   print("Running on dev")
   steps = 0
   totalLoss = 0
   numberOfWords = 0
   for dataPoint in held_out_data:
      steps += 1
   
      numberOfWords += len(dataPoint[comment_index])-1

      encoder_outputs, hidden = encoder.forward(torch.LongTensor(dataPoint[parent_index]).cuda(), None)
      target = torch.LongTensor(dataPoint[comment_index][1:]).view(-1).cuda()

      if not useAttention:
         output, _ = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]).cuda(), hidden)
      else:
         output, _, attentions = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]).view(-1,1).cuda(), hidden, encoder_outputs=encoder_outputs)
      loss = lossModuleNoAverage(output.view(-1, 10000+3), target.view(-1))

      crossEntropy = 0.99 * crossEntropy + (1-0.99) * loss.data.cpu().numpy()
      totalLoss +=  loss.data.cpu().numpy()
   devLosses.append(totalLoss/numberOfWords)
   print(devLosses)
   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
       print("Overfitting, stop")
       break


