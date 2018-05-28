import random
import torch
from utils import START_TOKEN, END_TOKEN

def train_example(input, output, language, encoder, decoder, encoder_optimizer, decoder_optimizer, params, loss_fn):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input.size(0)
    target_length = output.size(0)
    loss = 0.0

    encoder_outs = torch.zeros(params.max_length, encoder.hidden_size)
    for word in range(input_length):
        out, encoder_hidden = encoder(input[word], encoder_hidden)
        encoder_outs[word] = out[0, 0]

    # Whether or not the decoder uses the previous ground truth or decoded output in the next step.
    teacher_forcing = True if random.random() < params.teacher_forcing else False

    decoder_hidden = encoder_hidden
    decoder_in = torch.tensor([[START_TOKEN]])
    # Assuming we're using the attention decoder here...
    for word in range(target_length):
        decoder_out, decoder_hidden, _ = decoder(decoder_in, decoder_hidden, encoder_outs)
        loss += loss_fn(decoder_out, output[word])
        if teacher_forcing:
            decoder_in = output[word]
        else:
            vals, indices = topk(decoder_out, 1)
            decoder_in = indices.squeeze().detach()

        if decoder_in.item() == END_TOKEN:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def run_training_loop(params, encoder, decoder):
    lr = params.learning_rate
    n_iters = params.n_iters
