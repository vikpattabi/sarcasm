import torch.nn as nn
import pandas as pd

class embeddingModel(nn.Module):
    def __init__(self, embedding_size, vocabulary, num_samples, lr_in):
        super(embeddingModel, self).__init__()

        # The input is D where D is the word vector size
        self.embedding_size = embedding_size
        self.vocabulary = vocabulary
        self.embedding = torch.ones(self.embedding_size)
        self.num_samples = num_samples


    # input -> [1, D] user embedding
    def forward(self, word_embedding):
        relu_in = 0.0
        for embed in sampleFromVocab(self.num_samples):
            relu_in += (1 - torch.dot(word_embedding, self.embedding) + torch.dot(embed, self.embedding))
        # hidden is now a
        output = functional.relu(relu_in)
        return output

    def sampleFromVocab(num_to_sample):
        pass
