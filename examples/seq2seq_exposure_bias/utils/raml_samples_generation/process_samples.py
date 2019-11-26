from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sys
import re
import argparse
import torch
from util import read_corpus
import numpy as np
from scipy.misc import comb
from vocab import Vocab, VocabEntry
import math
from rouge import Rouge


def is_valid_sample(sent):
    tokens = sent.split(' ')
    return len(tokens) >= 1 and len(tokens) < 50


def sample_from_model(args):
    para_data = args.parallel_data
    sample_file = args.sample_file
    output = args.output

    tgt_sent_pattern = re.compile(r"^\[(\d+)\] (.*?)$")
    para_data = [l.strip().split(' ||| ') for l in open(para_data)]

    f_out = open(output, 'w')
    f = open(sample_file)
    f.readline()
    for src_sent, tgt_sent in para_data:
        line = f.readline().strip()
        assert line.startswith('****')
        line = f.readline().strip()
        print(line)
        assert line.startswith('target:')

        tgt_sent2 = line[len('target:'):]
        assert tgt_sent == tgt_sent2

        line = f.readline().strip()  # samples

        tgt_sent = ' '.join(tgt_sent.split(' ')[1:-1])
        tgt_samples = set()
        for i in range(1, 101):
            line = f.readline().rstrip('\n')
            m = tgt_sent_pattern.match(line)

            assert m, line
            assert int(m.group(1)) == i

            sampled_tgt_sent = m.group(2).strip()

            if is_valid_sample(sampled_tgt_sent):
                tgt_samples.add(sampled_tgt_sent)

        line = f.readline().strip()
        assert line.startswith('****')

        tgt_samples.add(tgt_sent)
        tgt_samples = list(tgt_samples)

        assert len(tgt_samples) > 0

        tgt_ref_tokens = tgt_sent.split(' ')
        bleu_scores = []
        for tgt_sample in tgt_samples:
            bleu_score = sentence_bleu([tgt_ref_tokens], tgt_sample.split(' '))
            bleu_scores.append(bleu_score)

        tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: bleu_scores[i], reverse=True)

        print('%d samples' % len(tgt_samples))

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_samples), file=f_out)
        for i in tgt_ranks:
            print('%s ||| %f' % (tgt_samples[i], bleu_scores[i]), file=f_out)
        print('*' * 50, file=f_out)

    f_out.close()


def get_new_ngram(ngram, n, vocab):
    """
    replace ngram `ngram` with a newly sampled ngram of the same length
    """

    new_ngram_wids = [np.random.randint(3, len(vocab)) for i in range(n)]
    new_ngram = [vocab.id2word[wid] for wid in new_ngram_wids]

    return new_ngram


def sample_ngram(args):
    src_sents = read_corpus(args.src, 'src')
    tgt_sents = read_corpus(args.tgt, 'src')  # do not read in <s> and </s>
    f_out = open(args.output, 'w')

    vocab = torch.load(args.vocab)
    tgt_vocab = vocab.tgt

    smooth_bleu = args.smooth_bleu
    sm_func = None
    if smooth_bleu:
        sm_func = SmoothingFunction().method3

    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        src_sent = ' '.join(src_sent)

        tgt_len = len(tgt_sent)
        tgt_samples = []
        tgt_samples_distort_rates = []    # how many unigrams are replaced

        # generate 100 samples

        # append itself
        tgt_samples.append(tgt_sent)
        tgt_samples_distort_rates.append(0)

        for sid in range(args.sample_size - 1):
            n = np.random.randint(1, min(tgt_len, args.max_ngram_size + 1))  # we do not replace the last token: it must be a period!

            idx = np.random.randint(tgt_len - n)
            ngram = tgt_sent[idx: idx + n]
            new_ngram = get_new_ngram(ngram, n, tgt_vocab)

            sampled_tgt_sent = list(tgt_sent)
            sampled_tgt_sent[idx: idx + n] = new_ngram

            # compute the probability of this sample
            # prob = 1. / args.max_ngram_size * 1. / (tgt_len - 1 + n) * 1 / (len(tgt_vocab) ** n)

            tgt_samples.append(sampled_tgt_sent)
            tgt_samples_distort_rates.append(n)

        # compute bleu scores or edit distances and rank the samples by bleu scores
        rewards = []
        for tgt_sample, tgt_sample_distort_rate in zip(tgt_samples, tgt_samples_distort_rates):
            if args.reward == 'bleu':
                reward = sentence_bleu([tgt_sent], tgt_sample, smoothing_function=sm_func)
            elif args.reward == 'rouge':
                rouge = Rouge()
                scores = rouge.get_scores(hyps=[' '.join(tgt_sample).decode('utf-8')], refs=[' '.join(tgt_sent).decode('utf-8')], avg=True)
                reward = sum([value['f'] for key, value in scores.items()])
            else:
                reward = -tgt_sample_distort_rate

            rewards.append(reward)

        tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: rewards[i], reverse=True)
        # convert list of tokens into a string
        tgt_samples = [' '.join(tgt_sample) for tgt_sample in tgt_samples]

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_samples), file=f_out)
        for i in tgt_ranks:
            print('%s ||| %f' % (tgt_samples[i], rewards[i]), file=f_out)
        print('*' * 50, file=f_out)

    f_out.close()


def sample_ngram_adapt(args):
    src_sents = read_corpus(args.src, 'src')
    tgt_sents = read_corpus(args.tgt, 'src')  # do not read in <s> and </s>
    f_out = open(args.output, 'w')

    vocab = torch.load(args.vocab)
    tgt_vocab = vocab.tgt

    max_len = max([len(tgt_sent) for tgt_sent in tgt_sents]) + 1

    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        src_sent = ' '.join(src_sent)

        tgt_len = len(tgt_sent)
        tgt_samples = []

        # generate 100 samples

        # append itself
        tgt_samples.append(tgt_sent)

        for sid in range(args.sample_size - 1):
            max_n = min(tgt_len - 1, 4)
            bias_n = int(max_n * tgt_len / max_len) + 1
            assert 1 <= bias_n <= 4, 'bias_n={}, not in [1,4], max_n={}, tgt_len={}, max_len={}'.format(bias_n, max_n, tgt_len, max_len)

            p = [1.0 / (max_n + 5)] * max_n
            p[bias_n - 1] = 1 - p[0] * (max_n - 1)
            assert abs(sum(p) - 1) < 1e-10, 'sum(p) != 1'

            n = np.random.choice(np.arange(1, int(max_n + 1)), p=p)  # we do not replace the last token: it must be a period!
            assert n < tgt_len, 'n={}, tgt_len={}'.format(n, tgt_len)

            idx = np.random.randint(tgt_len - n)
            ngram = tgt_sent[idx: idx + n]
            new_ngram = get_new_ngram(ngram, n, tgt_vocab)

            sampled_tgt_sent = list(tgt_sent)
            sampled_tgt_sent[idx: idx + n] = new_ngram

            tgt_samples.append(sampled_tgt_sent)

        # compute bleu scores and rank the samples by bleu scores
        bleu_scores = []
        for tgt_sample in tgt_samples:
            bleu_score = sentence_bleu([tgt_sent], tgt_sample)
            bleu_scores.append(bleu_score)

        tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: bleu_scores[i], reverse=True)
        # convert list of tokens into a string
        tgt_samples = [' '.join(tgt_sample) for tgt_sample in tgt_samples]

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_samples), file=f_out)
        for i in tgt_ranks:
            print('%s ||| %f' % (tgt_samples[i], bleu_scores[i]), file=f_out)
        print('*' * 50, file=f_out)

    f_out.close()


def sample_from_hamming_distance_payoff_distribution(args):
    src_sents = read_corpus(args.src, 'src')
    tgt_sents = read_corpus(args.tgt, 'src')  # do not read in <s> and </s>
    f_out = open(args.output, 'w')

    vocab = torch.load(args.vocab)
    tgt_vocab = vocab.tgt

    payoff_prob, Z_qs = generate_hamming_distance_payoff_distribution(max(len(sent) for sent in tgt_sents),
                                                                      vocab_size=len(vocab.tgt),
                                                                      tau=args.temp)

    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        tgt_samples = []  # make sure the ground truth y* is in the samples
        tgt_sent_len = len(tgt_sent) - 3  # remove <s> and </s> and ending period .
        tgt_ref_tokens = tgt_sent[1:-1]
        bleu_scores = []

        # sample an edit distances
        e_samples = np.random.choice(range(tgt_sent_len + 1), p=payoff_prob[tgt_sent_len], size=args.sample_size,
                                     replace=True)

        for i, e in enumerate(e_samples):
            if e > 0:
                # sample a new tgt_sent $y$
                old_word_pos = np.random.choice(range(1, tgt_sent_len + 1), size=e, replace=False)
                new_words = [vocab.tgt.id2word[wid] for wid in np.random.randint(3, len(vocab.tgt), size=e)]
                new_tgt_sent = list(tgt_sent)
                for pos, word in zip(old_word_pos, new_words):
                    new_tgt_sent[pos] = word

                bleu_score = sentence_bleu([tgt_ref_tokens], new_tgt_sent[1:-1])
                bleu_scores.append(bleu_score)
            else:
                new_tgt_sent = list(tgt_sent)
                bleu_scores.append(1.)

            # print('y: %s' % ' '.join(new_tgt_sent))
            tgt_samples.append(new_tgt_sent)


def generate_hamming_distance_payoff_distribution(max_sent_len, vocab_size, tau=1.):
    """compute the q distribution for Hamming Distance (substitution only) as in the RAML paper"""
    probs = dict()
    Z_qs = dict()
    for sent_len in range(1, max_sent_len + 1):
        counts = [1.]  # e = 0, count = 1
        for e in range(1, sent_len + 1):
            # apply the rescaling trick as in https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
            count = comb(sent_len, e) * math.exp(-e / tau) * ((vocab_size - 1) ** (e - e / tau))
            counts.append(count)

        Z_qs[sent_len] = Z_q = sum(counts)
        prob = [count / Z_q for count in counts]
        probs[sent_len] = prob

        # print('sent_len=%d, %s' % (sent_len, prob))

    return probs, Z_qs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sample_from_model', 'sample_ngram_adapt', 'sample_ngram'], required=True)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    parser.add_argument('--parallel_data', type=str)
    parser.add_argument('--sample_file', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--reward', choices=['bleu', 'edit_dist', 'rouge'], default='bleu')
    parser.add_argument('--max_ngram_size', type=int, default=4)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--smooth_bleu', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'sample_ngram':
        sample_ngram(args)
    elif args.mode == 'sample_from_model':
        sample_from_model(args)
    elif args.mode == 'sample_ngram_adapt':
        sample_ngram_adapt(args)
