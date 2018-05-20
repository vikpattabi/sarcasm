import torch.nn as nn

class embeddingModel(nn.module):
    def __init__(self, input_size, hidden_size, output_size, vocabulary):
        super(embeddingModel, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.vocabulary = vocabulary

    # input -> [1, D] user embedding
    def forward(self, input, hidden, user_vec):
        hidden = self.i2h(input)
        # hidden is now a
        output = self.relu()
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def sampleFromVocab():
        return 
