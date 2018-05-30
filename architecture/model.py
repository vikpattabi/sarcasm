import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH=50

def embeddings(vocab_size, embedding_size):
  return nn.Embedding(vocab_size, embedding_size)

class encoderRNN(nn.Module):
    def __init__(self, hidden_size=200, embedding_size=200, embeddings=None, dropout_p = 0.1):
        super(encoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
      
        self.embedding = embeddings
        self.gru = nn.GRU(embedding_size, hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), -1, self.embedding_size)
        output = self.dropout(embedded)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class decoderRNN(nn.Module):
    def __init__(self, hidden_size=200, embedding_size=200, embeddings=None, vocab_size=10000+3, dropout_p = 0.1):
        super(decoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
      
        self.embedding = embeddings
        self.gru = nn.GRU(embedding_size, hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(len(input), -1, self.embedding_size)

        output = self.dropout(embedded)

        #output = functional.relu(output)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class attentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size=200, embedding_size=200, embeddings=None, vocab_size=10000+3, dropout_p = 0.1):
        super(attentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
  #      self.max_length = max_length
        self.embedding_size = embedding_size

        self.embedding = embeddings

        self.attn_hidden_1 = nn.Linear(self.hidden_size, 100)
        self.attn_hidden_2 = nn.Linear(self.hidden_size, 100)
        self.attn_hidden_3 = nn.Linear(embedding_size, 100)

        self.attn = nn.Linear(100,1)
        self.attn.weight.data.fill_(0)

        self.attn_combine = nn.Linear(self.hidden_size + embedding_size, embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embedding_size+hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
#        self.out.weight.data.copy_(self.embedding.weight.data) # .transpose(0,1) # for this, we would need to match up embedding and hidden dimensions

    def forward(self, input, hidden, encoder_outputs):

        batchSize = input.size()[1]
        length_of_target = input.size()[0]

        length_of_source = encoder_outputs.size()[0]
        encoder_outputs = self.dropout(encoder_outputs.transpose(0,1)) # now: batchSize x length_of_source

        embedded = self.embedding(input).view(length_of_target, batchSize, self.embedding_size)
        embedded = self.dropout(embedded)


        attention_hidden_3 = self.attn_hidden_3(embedded) # length_of_target x batchSize x 100
        attention_hidden_2 = self.attn_hidden_2(encoder_outputs) # batchSize x length_of_source x 100

        outputs = []
        attentions = []
        for i in range(length_of_target):
           attention_hidden_1 = self.attn_hidden_1(hidden) # 1 x batchSize x 100 
           attention_hidden = F.relu(attention_hidden_1.squeeze(0).unsqueeze(1) + attention_hidden_2 + attention_hidden_3[i].unsqueeze(1)) # length_of_source x batchSize x 100
           attention_logits = self.attn(attention_hidden) 
#           print(encoder_outputs)

           #  batchSize x length_of_source x 1
           attention = F.softmax(attention_logits, dim=1).view(batchSize, 1, length_of_source)
#           print(attention)
           attn_applied = torch.bmm(attention, encoder_outputs) #.view(1,-1,self.hidden_size)) #.view(length_of_source, 1, self.hidden_size))
#           print(attn_applied)
#           print("..")
           output = self.dropout(torch.cat((embedded[i], attn_applied.squeeze(1)), 1).unsqueeze(0))
           output, hidden = self.gru(output, hidden)
           # output : 1 x batchSize x 200
  
           output = F.log_softmax(self.out(output.squeeze(0)), dim=1)
           outputs.append(output)
           attentions.append(attention)
        outputs = torch.cat([x.unsqueeze(0) for x in outputs], dim=0)
        return outputs, hidden, attentions

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



