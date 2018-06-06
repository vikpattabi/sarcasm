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

subreddit_index = read.keys.index("subreddit")



# , subreddit_embeddings=None, stoi_subreddits=None, itos_subreddits=None
def run_training_loop(training_data, held_out_data, encoder, decoder, embeddings, batchSize=32, learning_rate=0.001, optimizer="Adam", useAttention=False, stoi=None, itos=None, subreddit_embeddings=None, itos_subreddits=None, stoi_subreddits=None, args=None):

 quotation_mark_index = stoi.get('"', -1) + 3

 def predictFromInput(input_sentence, subreddit):
    input = read.encode_sentence(input_sentence, stoi)
    
    encoder_outputs, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
    subreddit = torch.LongTensor([stoi_subreddits.get(subreddit, -1)+1]).cuda()

    generated = [torch.LongTensor([0]).cuda()]
    generated_words = []
    while True:
       input = generated[-1]
       if not useAttention:
          output, hidden = decoder.forward(input, hidden, subreddits=subreddit)
       else:
          output, hidden, attention = decoder.forward(input.view(1,1), hidden, encoder_outputs=encoder_outputs, subreddits=subreddit)
          print(attention[0].view(-1).data.cpu().numpy()[:])
       _, predicted = torch.topk(output, 3, dim=2)
       predicted = predicted.data.cpu().view(3).numpy()
       for i in range(3):
          if predicted[i] != 2 and predicted[i] != quotation_mark_index:
            predicted = predicted[i]
            break
       
       predicted_numeric = predicted
       if predicted_numeric == 1 or predicted_numeric == 0 or len(generated_words) > 100:
          return " ".join(generated_words)
       elif predicted_numeric ==2:
         generated_words.append("OOV")
       else:
         generated_words.append(itos[predicted_numeric-3])
       generated.append(torch.LongTensor([predicted_numeric]).cuda())


 
 def discriminivativeDecoding(input_sentence, subreddit1, subreddit2):
    input = read.encode_sentence(input_sentence, stoi)
    
    encoder_outputs, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
    subreddit1 = torch.LongTensor([stoi_subreddits.get(subreddit1, -1)+1]).cuda()
    subreddit2 = torch.LongTensor([stoi_subreddits.get(subreddit2, -1)+1]).cuda()
   
    beamSize = 10

    sampling= True

    finished = []
    generated = [[(0, 0.0, False)]]
#    hidden = hidden.expand(1, len(generated), 200).contiguous()
    hidden2 = hidden
    while len(generated) > 0:
       input = torch.LongTensor([x[-1][0] for x in generated]).view(1,-1).cuda()
       encoder_outputs_expanded = encoder_outputs.expand(-1,len(generated), -1)
       if not useAttention:
          output, hidden = decoder.forward(input, hidden, subreddits=subreddit1)
          output2, hidden2 = decoder.forward(input, hidden2, subreddits=subreddit2)
       else:
          output, hidden, attention = decoder.forward(input, hidden, encoder_outputs=encoder_outputs_expanded, subreddits=subreddit1)
          output2, hidden2, attention2 = decoder.forward(input, hidden2, encoder_outputs=encoder_outputs_expanded, subreddits=subreddit2)
#          print(attention[0].view(-1).data.cpu().numpy()[:])

#       if len(generated) == 1:
#          hidden = hidden.expand(1, beamSize-len(finished), 200).contiguous()
#          hidden2 = hidden2.expand(1, beamSize-len(finished), 200).contiguous()

       hiddenStates = [hidden.squeeze(0)[j] for j in range(len(generated))]
       hiddenStates2 = [hidden2.squeeze(0)[j] for j in range(len(generated))]

       topk = beamSize+3 if len(generated) == 1 else (10 if sampling else 3)
       probabilities, predicted = torch.topk(output, topk, dim=2) # output2

       predicted = predicted.data.cpu().view(len(generated), topk).numpy()
       probabilities = probabilities.data.cpu().view(len(generated), topk).numpy()


       newVersions = []
       for j in range(len(generated)):
         for i in range(topk):
            if predicted[j][i] != 2 and predicted[j][i] != quotation_mark_index:
               newVersions.append((generated[j] + [(predicted[j][i], probabilities[j][i] + generated[j][-1][1], generated[j][-1][2])], j    ))

       if not sampling:
          newVersions = sorted(newVersions, key=lambda x:x[0][-1][1], reverse=True)
       else:
          random.shuffle(newVersions)



       generated = [x[0] for x in newVersions[:(beamSize - len(finished))]]


#       print(generated)
#       quit()

       allHaveFinished = True
       assert len(generated) + len(finished) <= beamSize
       for j in range(len(generated)):
           assert j < len(generated)
           assert len(generated[j]) > 1
           assert len(generated[j][-1]) > 1
           if not (generated[j][-1][0] == 1 or generated[j][-1][0] == 0 or len(generated[j]) > 100):
               allHaveFinished=False
           else:
               finished.append(generated[j])
               generated[j] = False
       if allHaveFinished:
          break

       hidden = torch.cat([hiddenStates[newVersions[i][1]].unsqueeze(0) for i in range(len(generated)) if generated[i] is not False], dim=0).unsqueeze(0)
       hidden2 = torch.cat([hiddenStates2[newVersions[i][1]].unsqueeze(0) for i in range(len(generated)) if generated[i] is not False], dim=0).unsqueeze(0)



       generated = [x for x in generated if x is not False]

#       print((hidden.size(), hidden2.size(), len(generated)))

 #   print(finished[0])
    for j in range(beamSize):
        string = ""
        for word in finished[j][1:]:
#            print(word)
            if word[0] == 1 or word[0] == 0:
                break
            string+=" "+itos[word[0]-3]
        print(string)
        print(finished[j][-1][1])


#    input = read.encode_sentence(input_sentence, stoi)
#    
#    encoder_outputs, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
#
#
#    targets = torch.LongTensor([[0] + [x[0] for x in sentence] for sentence in generated]).transpose(0,1).cuda()
#    
#    subreddit2 = torch.LongTensor([stoi_subreddits.get(subreddit2, -1)+1]).cuda()
#
#    hidden = hidden.expand(1, len(generated), 200).contiguous()
#
#    if not useAttention:
#       output, hidden = decoder.forward(targets[:-1], hidden, subreddits=subreddit2)
#    else:
#       output, hidden, attention = decoder.forward(targets[:-1], hidden, encoder_outputs=encoder_outputs, subreddits=subreddit2)
#       print(attention[0].view(-1).data.cpu().numpy()[:])
#    probabilities2 = 



#    predicted_numeric = predicted
#    if predicted_numeric == 1 or predicted_numeric == 0 or len(generated_words) > 100:
#       return " ".join(generated_words)
#    elif predicted_numeric ==2:
#      generated_words.append("OOV")
#    else:
#      generated_words.append(itos[predicted_numeric-3])
#    generated.append(torch.LongTensor([predicted_numeric]).cuda())

    return "" 
 

 encoder_optimizer = None
 decoder_optimizer = None
 embeddings_optimizer = None
 subreddit_embeddings_optimizer = None
 
 optimizer = "Adam"
 
 
 if optimizer == 'Adam':
     encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
     decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
     embeddings_optimizer = torch.optim.Adam(embeddings.parameters(), lr = learning_rate)
     subreddit_embeddings_optimizer = torch.optim.Adam(subreddit_embeddings.parameters(), lr = learning_rate)

 else:
     encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr = learning_rate)
     decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = learning_rate)
     embeddings_optimizer = torch.optim.SGD(embeddings.parameters(), lr = learning_rate)
     subreddit_embeddings_optimizer = torch.optim.SGD(subreddit_embeddings.parameters(), lr = learning_rate)

 lossModule = torch.nn.NLLLoss(ignore_index=0)
 lossModuleNoAverage = torch.nn.NLLLoss(size_average=False, ignore_index=0)




 if args.load_from is not None:
   checkpoint = torch.load("data/checkpoints/"+args.load_from+".pth.tar")
   subreddit_embeddings.load_state_dict(checkpoint["subreddit_embeddings"])
   embeddings.load_state_dict(checkpoint["embeddings"])
   encoder.load_state_dict(checkpoint["encoder"])
   decoder.load_state_dict(checkpoint["decoder"])
   encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
   decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
   embeddings_optimizer.load_state_dict(checkpoint["embeddings_optimizer"])
   subreddit_embeddings_optimizer.load_state_dict(checkpoint["subreddit_embeddings_optimizer"])
            


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
      subreddit_embeddings_optimizer.zero_grad()
  

      context_sentence = collectAndPadInput(current, parent_index)
      response_sentence = collectAndPadInput(current, comment_index)

      subreddits = torch.LongTensor([stoi_subreddits.get(x[subreddit_index], -1)+1 for x in current]).cuda()

      encoder_outputs, hidden = encoder.forward(torch.LongTensor(context_sentence).cuda(), None)

      response = torch.LongTensor(response_sentence).cuda()
     

      if not useAttention:
         output, _ = decoder.forward(response[:-1], hidden, subreddits=subreddits)
      else:
         output, _, attentions = decoder.forward(response[:-1], hidden, encoder_outputs=encoder_outputs, subreddits=subreddits)

      loss = lossModule(output.view(-1, 10000+3), response[1:].view(-1))


      crossEntropy = 0.99 * crossEntropy + (1-0.99) * loss.data.cpu().numpy()
   
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()
      embeddings_optimizer.step()
      if not args.freeze_subreddit_embeddings:
          subreddit_embeddings_optimizer.step()
  
   
      if steps % 1000 == 0: # 
#          encoder.train(False)
#          decoder.train(False)

          print((epoch,steps,crossEntropy))
          print(devLosses)
          print("worldnews")
          print(predictFromInput(["This", "article", "is", "awesome", "."], "worldnews"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "worldnews"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "worldnews"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "worldnews"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "worldnews"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "worldnews"))
 
          print("funny")
          print(predictFromInput(["This", "article", "is", "awesome", "."], "funny"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "funny"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "funny"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "funny"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "funny"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "funny"))


          print("gaming")
          print(predictFromInput(["This", "article", "is", "awesome", "."], "gaming"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "gaming"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "gaming"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "gaming"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "gaming"))
          print(predictFromInput(["This", "article", "is", "awesome", "."], "gaming"))


          print("worldnews")
          print(discriminivativeDecoding(["This", "article", "is", "awesome", "."], "worldnews", "funny"))
          print("funny")
          print(discriminivativeDecoding(["This", "article", "is", "awesome", "."], "funny", "worldnews"))

#          encoder.train(True)
#          decoder.train(True)


          # save current model
#          quit()

#          print(predictFromInput(["This", "article", "is", "awesome", "."]))
 #         print(predictFromInput(["Bankers", "celebrate", "the", "start", "of", "the", "Trump", "era", "."]))

   # At the end of every epoch, we run on the development partition and record the log-likelihood. As soon as it drops, we stop training
   print("Running on dev")
   print(args)
   encoder.train(False)
   decoder.train(False)
   steps = 0
   totalLoss = 0
   numberOfWords = 0
   for dataPoint in held_out_data:
      steps += 1
   
      numberOfWords += len(dataPoint[comment_index])-1


      subreddits = torch.LongTensor([stoi_subreddits.get(x[subreddit_index], -1)+1 for x in [dataPoint]]).cuda()


      encoder_outputs, hidden = encoder.forward(torch.LongTensor(dataPoint[parent_index]).cuda(), None)
      target = torch.LongTensor(dataPoint[comment_index][1:]).view(-1).cuda()

      if not useAttention:
         output, _ = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]).cuda(), hidden, subreddits=subreddits)
      else:
         output, _, attentions = decoder.forward(torch.LongTensor(dataPoint[comment_index][:-1]).view(-1,1).cuda(), hidden, encoder_outputs=encoder_outputs, subreddits=subreddits)
      loss = lossModuleNoAverage(output.view(-1, 10000+3), target.view(-1))

      crossEntropy = 0.99 * crossEntropy + (1-0.99) * loss.data.cpu().numpy()
      totalLoss +=  loss.data.cpu().numpy()
   devLosses.append(totalLoss/numberOfWords)
   print(devLosses)
   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
       print("Overfitting, stop")
       return

   if args.save_to is not None:
      torch.save({"subreddit_embeddings_optimizer" : subreddit_embeddings_optimizer.state_dict(), "subreddit_embeddings" : subreddit_embeddings.state_dict(), "embeddings" : embeddings.state_dict(), "encoder" : encoder.state_dict(), "decoder" : decoder.state_dict(), "encoder_optimizer" : encoder_optimizer.state_dict(), "decoder_optimizer" : decoder_optimizer.state_dict(),"embeddings_optimizer" : embeddings_optimizer.state_dict()}, "data/checkpoints/"+args.save_to+".pth.tar")


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

