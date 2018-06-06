import torch
#from utils import START_TOKEN, END_TOKEN

from data import read

# Contains functions for evaluating our results, using both the BLEU metric and hand-evaluation.

# CITE: From here - https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py

comment_index = read.keys.index("comment")
parent_index = read.keys.index("parent_comment")
subreddit_index = read.keys.index("subreddit")

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


# , subreddit_embeddings=None, stoi_subreddits=None, itos_subreddits=None
def run_test(test_data, encoder, decoder, embeddings, stoi=None, itos=None, subreddit_embeddings=None, itos_subreddits=None, stoi_subreddits=None, useAttention=False):
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
           elif predicted_numeric ==2:
             generated_words.append("OOV")
           else:
             generated_words.append(itos[predicted_numeric-3])
           generated.append(torch.LongTensor([predicted_numeric]).cuda())

    for sample in test_data:
       parent_comment = []
       ground_truth_response = []
       parent = sample[parent_index]
       comment = sample[comment_index]
       subreddit = sample[subreddit_index]

       for word in parent:
           original.append(itos[word] - 3)
       for word in comment:
           ground_truth_response.append(itos[word] - 3)

       output = predictFromInput(original, subreddit)
       output = output.split(" ")
       net_bleu += bleu(bleu_stats(output, ground_truth_response))
       num_samples_tested += 1

    print('------------------')
    print('Mean BLEU score: %.2f' (net_bleu / num_samples_tested))
    print('Number of test samples: %d' (num_samples_tested))
