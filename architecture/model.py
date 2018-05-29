import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH=50

def embeddings(vocab_size, embedding_size):
  return nn.Embedding(vocab_size, embedding_size)

class encoderRNN(nn.Module):
    def __init__(self, hidden_size=200, embedding_size=200, embeddings=None):
        super(encoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
       
        self.embedding = embeddings
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), -1, self.embedding_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class decoderRNN(nn.Module):
    def __init__(self, hidden_size=200, embedding_size=200, embeddings=None, vocab_size=10000+3):
        super(decoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
       
        self.embedding = embeddings
        self.gru = nn.GRU(embedding_size, hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)
#        self.out.data = self.embedding.data.transpose(0,1)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), -1, self.embedding_size)
        #output = functional.relu(output)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class attentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size=200, embedding_size=200, embeddings=None, vocab_size=10000+3):
        super(attentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
 #       self.dropout_p = dropout_p
  #      self.max_length = max_length
        self.embedding_size = embedding_size

        self.embedding = embeddings

        self.attn_hidden_1 = nn.Linear(self.hidden_size, 100)
        self.attn_hidden_2 = nn.Linear(self.hidden_size, 100)
        self.attn_hidden_3 = nn.Linear(embedding_size, 100)

        self.attn = nn.Linear(100,1)
        self.attn.weight.data.fill_(0)

        self.attn_combine = nn.Linear(self.hidden_size + embedding_size, embedding_size)
#        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embedding_size+hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
#        self.out.weight.data.copy_(self.embedding.weight.data) # .transpose(0,1) # for this, we would need to match up embedding and hidden dimensions

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(len(input), 1, self.embedding_size)
#        embedded = self.dropout(embedded)

        length_of_source = encoder_outputs.size()[0]

        attention_hidden_3 = self.attn_hidden_3(embedded)
     #   print(attention_hidden_3.size())
        attention_hidden_2 = self.attn_hidden_2(encoder_outputs)
    #    print(attention_hidden_2.size())

        outputs = []
        attentions = []
        for i in range(len(input)):
          
#           print(encoder_outputs.size())
           attention_hidden_1 = self.attn_hidden_1(hidden.view(1,-1)).unsqueeze(0)
   #        print(attention_hidden_1.size())

           attention_hidden = F.relu(attention_hidden_1 + attention_hidden_2 + attention_hidden_3[i].unsqueeze(0))
           attention_logits = self.attn(attention_hidden) #.view(length_of_source, 1).view(-1) # TODO for minibatching, will we need to transpose this before softmaxing?
  #         print(attention_logits.size())

           attention = F.softmax(attention_logits).view(1,1,-1)
#           print(attention)
#           print(combined.size())
#           print(encoder_outputs.size())

           attn_applied = torch.bmm(attention, encoder_outputs.view(1,-1,self.hidden_size)) #.view(length_of_source, 1, self.hidden_size))
 #          print(attn_applied.size())
 
           output = torch.cat((embedded[i], attn_applied.squeeze(0)), 1).unsqueeze(0)
#           output = self.attn_combine(output).unsqueeze(0)
#   
#           output = F.relu(output)
           output, hidden = self.gru(output, hidden)
   
           output = F.log_softmax(self.out(output[0]), dim=1)
           outputs.append(output)
           attentions.append(attention)
     #      quit()
        return outputs, hidden, attentions

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



