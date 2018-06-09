import torch
#from utils import START_TOKEN, END_TOKEN

from data import read

# Contains functions for evaluating our results, using both the BLEU metric and hand-evaluation.

# CITE: From here - https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py

comment_index = read.keys.index("comment")
parent_index = read.keys.index("parent_comment")
subreddit_index = read.keys.index("subreddit")

from collections import Counter
import math

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 2):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


# , subreddit_embeddings=None, stoi_subreddits=None, itos_subreddits=None
def run_test(test_data, encoder, decoder, embeddings, useAttention=False, stoi=None, itos=None, subreddit_embeddings=None, itos_subreddits=None, stoi_subreddits=None, args=None):
    net_bleu = 0.0
    num_samples_tested = 0

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
           elif predicted_numeric == 2:
             generated_words.append("OOV")
           else:
             generated_words.append(itos[predicted_numeric-3])
           generated.append(torch.LongTensor([predicted_numeric]).cuda())


    def simpleBeamSearch(input_sentence, subreddit1):
       input = read.encode_sentence(input_sentence, stoi)
       
       encoder_outputs, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
       subreddit1 = torch.LongTensor([stoi_subreddits.get(subreddit1, -1)+1]).cuda()
      
       beamSize = 10
   
       sampling= False
   
       groups = 1
   
       finished = [[] for _ in range(groups)] 
       generated = [[[(0, 0.0, False)]] for _ in range(groups)]
       hidden = [hidden for _ in range(groups)]
       allHaveFinished = [False for _ in range(groups)]
       while len(generated) > 0:
         if all(allHaveFinished):
             break
         for group in range(groups):
          if allHaveFinished[group]:
             continue
   #       print(generated[group])
          input = torch.LongTensor([x[-1][0] for x in generated[group]]).view(1,-1).cuda()
          encoder_outputs_expanded = encoder_outputs.expand(-1,len(generated[group]), -1)
          output, hidden[group] = decoder.forward(input, hidden[group], subreddits=subreddit1)
   
    #      print(output)
   
   
          hiddenStates = [hidden[group].squeeze(0)[j] for j in range(len(generated[group]))]
   
   
   
          topk = beamSize+3 if len(generated[group]) == 1 else (10 if sampling else 3)
          probabilities, predicted = torch.topk(output, topk, dim=2) # output2
   
   
          predicted = predicted.data.cpu().view(len(generated[group]), topk).numpy()
          probabilities = probabilities.data.cpu().view(len(generated[group]), topk).numpy()
   
   
          newVersions = []
          for j in range(len(generated[group])):
            for i in range(topk):
               if predicted[j][i] != 2 and predicted[j][i] != quotation_mark_index:
                  newVersions.append((generated[group][j] + [(predicted[j][i], probabilities[j][i] + generated[group][j][-1][1], generated[group][j][-1][2])], j    ))
   
          if not sampling:
             newVersions = sorted(newVersions, key=lambda x:x[0][-1][1], reverse=True)
          else:
             random.shuffle(newVersions)
   
   
          generated[group] = [x[0] for x in newVersions[:(beamSize - len(finished[group]))]]
   
          allHaveFinished[group] = True
          assert len(generated[group]) + len(finished[group]) <= beamSize
          for j in range(len(generated[group])):
              assert j < len(generated[group])
              assert len(generated[group][j]) > 1
              assert len(generated[group][j][-1]) > 1
              if not (generated[group][j][-1][0] == 1 or generated[group][j][-1][0] == 0 or len(generated[group][j]) > 100):
                  allHaveFinished[group]=False
              else:
                  finished[group].append(generated[group][j])
                  generated[group][j] = False
          if not allHaveFinished[group]:
            hidden[group] = torch.cat([hiddenStates[newVersions[i][1]].unsqueeze(0) for i in range(len(generated[group])) if generated[group][i] is not False], dim=0).unsqueeze(0)
   
          generated[group] = [x for x in generated[group] if x is not False]
   
   
   
       for group in range(groups): 
         print(group)
         for j in range(len(finished)):
             string = ""
             for word in finished[group][j][1:]:
     #            print(word)
                 if word[0] == 1 or word[0] == 0:
                     break
                 string+=" "+itos[word[0]-3]
             print(string)
             print(finished[group][j][-1][1])
       return "" 




    def discriminivativeDecoding(input_sentence, subreddit1):
       input = read.encode_sentence(input_sentence, stoi)
       
       encoder_outputs, hidden = encoder.forward(torch.LongTensor(input).cuda(), None)
       subreddit1 = torch.LongTensor([stoi_subreddits.get(subreddit1, -1)+1]).cuda()
      
       beamSize = 10
   
       sampling= False
   
       groups = 10
   
       finished = [[] for _ in range(groups)] 
       generated = [[[(0, 0.0, False, 0.0)]] for _ in range(groups)]
       hidden = [hidden for _ in range(groups)]
       allHaveFinished = [False for _ in range(groups)]
       counters = [Counter() for _ in range(groups)]
       while len(generated) > 0:
         if all(allHaveFinished):
             break
         for group in range(groups):
           if allHaveFinished[group]:
              continue
           input = torch.LongTensor([x[-1][0] for x in generated[group]]).view(1,-1).cuda()
           encoder_outputs_expanded = encoder_outputs.expand(-1,len(generated[group]), -1)
           output, hidden[group] = decoder.forward(input, hidden[group], subreddits=subreddit1)
    
           hiddenStates = [hidden[group].squeeze(0)[j] for j in range(len(generated[group]))]
    
           topk = beamSize+3 # if len(generated[group]) == 1 else (10 if sampling else 3)
           probabilities, predicted = torch.topk(output, topk, dim=2) # output2
           
    
           predicted = predicted.data.cpu().view(len(generated[group]), topk).numpy()
           probabilities = probabilities.data.cpu().view(len(generated[group]), topk).numpy()
#           print(predicted)
#           print(probabilities)
   
           newVersions = []
           for j in range(len(generated[group])):
             for i in range(topk):
                if predicted[j][i] != 2 and predicted[j][i] != quotation_mark_index:
                   similarityPenalty = 0.3 * counters[group].get(predicted[j][i], 0)

                   newVersions.append((generated[group][j] + [(predicted[j][i], probabilities[j][i] - similarityPenalty + generated[group][j][-1][1], generated[group][j][-1][2], probabilities[j][i] + generated[group][j][-1][3])], j    ))
    
           if not sampling:
              newVersions = sorted(newVersions, key=lambda x:x[0][-1][1], reverse=True)
           else:
              random.shuffle(newVersions)
    
    
           generated[group] = [x[0] for x in newVersions[:(beamSize - len(finished[group]))]]
    
           allHaveFinished[group] = True
           assert len(generated[group]) + len(finished[group]) <= beamSize
           for j in range(len(generated[group])):
               assert j < len(generated[group])
               assert len(generated[group][j]) > 1
               assert len(generated[group][j][-1]) > 1
               if not (generated[group][j][-1][0] == 1 or generated[group][j][-1][0] == 0 or len(generated[group][j]) > 100):
                   allHaveFinished[group]=False
               else:
                   finished[group].append(generated[group][j])
                   generated[group][j] = False
           if not allHaveFinished[group]:
             hidden[group] = torch.cat([hiddenStates[newVersions[i][1]].unsqueeze(0) for i in range(len(generated[group])) if generated[group][i] is not False], dim=0).unsqueeze(0)
    
           generated[group] = [x for x in generated[group] if x is not False]
           for group2 in range(group, groups):
             for new in generated[group]:
              counters[group2].update([new[-1][0]])
#           print(counters[group]) 
  
       results = [] 
       for group in range(groups): 
#         print(group)
         for j in range(len(finished)):
             string = ""
             wordsInString = 0
             for word in finished[group][j][1:]:
     #            print(word)
                 if word[0] == 1 or word[0] == 0:
                     break
                 wordsInString += 1
                 string+=" "+itos[word[0]-3]
             if j == 0:
                results.append((string, finished[group][j][-1][3]))


             break
#             print(string)
#             print(finished[group][j][-1][1])
#             print(finished[group][j][-1][3])
#             print(finished[group][j][-1][3]/wordsInString)

       results = sorted(results, key=lambda x:x[1], reverse=True)
       for result in results:
          print(result[0])
       return "" 




    if args.load_from is not None:
        checkpoint = torch.load("data/checkpoints/"+args.load_from+".pth.tar")
        subreddit_embeddings.load_state_dict(checkpoint["subreddit_embeddings"])
        embeddings.load_state_dict(checkpoint["embeddings"])
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
   



    counter = 0
    for sample in test_data:
       counter += 1
       if counter % 100 == 0:
         print(counter)
         print(float(net_bleu) / num_samples_tested)
       parent_comment = []
       ground_truth_response = []
       parent = sample[parent_index]
       comment = sample[comment_index]
       subreddit = sample[subreddit_index]
       original = []

       for word in parent[1:-1]:
           original.append(itos[word-3] if word > 2 else "OOV")
       for word in comment[1:-1]:
           ground_truth_response.append(itos[word-3] if word > 2 else "OOV")

        
       output = predictFromInput(original, subreddit)
       
       output = output.split(" ")
       if not args.bleu_only:
           print(subreddit)
           print(original)
           print(output)
           encoder.train(False)
           decoder.train(False)
           print("Deterministic Greedy Decoding")
           print(predictFromInput(original, subreddit))
           encoder.train(True)
           decoder.train(True)
           print("Decoding with Dropout")
           print(predictFromInput(original, subreddit))
           print(predictFromInput(original, subreddit))
           print(predictFromInput(original, subreddit))
           print(predictFromInput(original, subreddit))
           print(predictFromInput(original, subreddit))
           print(predictFromInput(original, subreddit))
           print(predictFromInput(original, subreddit))
           encoder.train(False)
           decoder.train(False)
           print("Diverse Decoding")
           discriminivativeDecoding(original, subreddit)
           encoder.train(True)
           decoder.train(True)

           print(ground_truth_response)
       new_bleu = bleu(bleu_stats(output, ground_truth_response))
       net_bleu += new_bleu
       num_samples_tested += 1

    print('------------------')
    print('Mean BLEU score: '+str(float(net_bleu) / num_samples_tested))
    print('Number of test samples: '+str(num_samples_tested))

