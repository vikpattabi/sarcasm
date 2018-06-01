import random
import torch
#from utils import START_TOKEN, END_TOKEN

from data import read

def collectAndPadInput(current, index):
      maximumAllowedLength = 50
#      current = [x[:] for x in current]
      maxLength = max([len(x[index]) for x in current])
      context_sentence = []
      for i in range(maxLength):
         context_sentence.append([x[index][i] if i < len(x[index]) else 0 for x in current])
      return context_sentence


comment_index = read.keys.index("comment")
parent_index = read.keys.index("parent_comment")





def run_training_loop(training_data, held_out_data, encoder, decoder, embeddings, batchSize=32, learning_rate=0.001, optimizer="Adam", useAttention=False, stoi=None, itos=None, subreddit_embeddings=None, stoi_subreddits=None, itos_subreddits=None):


 def predictFromInput(input_sentence):
    input = read.encode_sentence(input_sentence, stoi)
    
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
 
 lossModule = torch.nn.NLLLoss(ignore_index=0)
 lossModuleNoAverage = torch.nn.NLLLoss(size_average=False, ignore_index=0)



 devLosses = []

 batchSize = 32

 training_partitions = list(range(int(len(training_data)/batchSize)))



 for epoch in range(1000):

   random.shuffle(training_partitions)

   encoder.train(True) # set to training mode (make sure dropout is turned on again after running on dev set)
   decoder.train(True)
  
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

   # At the end of every epoch, we run on the development partition and record the log-likelihood. As soon as it drops, we stop training
   print("Running on dev")
   encoder.train(False)
   decoder.train(False)
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
       return



#def train_example(input, output, encoder, decoder, encoder_optimizer, decoder_optimizer, params, loss_fn):
#    encoder_hidden = encoder.initHidden()
#    encoder_optimizer.zero_grad()
#    decoder_optimizer.zero_grad()
#
#    input_length = input.size(0)
#    target_length = output.size(0)
#    loss = 0.0
#
#    encoder_outs = torch.zeros(params.max_length, encoder.hidden_size)
#    for word in range(input_length):
#        out, encoder_hidden = encoder(input[word], encoder_hidden)
#        encoder_outs[word] = out[0, 0]
#
#    # Whether or not the decoder uses the previous ground truth or decoded output in the next step.
#    teacher_forcing = True if random.random() < params.teacher_forcing else False
#
#    decoder_hidden = encoder_hidden
#    decoder_in = torch.tensor([[START_TOKEN]])
#    # Assuming we're using the attention decoder here...
#    for word in range(target_length):
#        decoder_out, decoder_hidden, _ = decoder(decoder_in, decoder_hidden, encoder_outs)
#        loss += loss_fn(decoder_out, output[word])
#        if teacher_forcing:
#            decoder_in = output[word]
#        else:
#            vals, indices = topk(decoder_out, 1)
#            decoder_in = indices.squeeze().detach()
#
#        if decoder_in.item() == END_TOKEN:
#            break
#
#    loss.backward()
#    encoder_optimizer.step()
#    decoder_optimizer.step()
#    return loss.item() / target_length
#

