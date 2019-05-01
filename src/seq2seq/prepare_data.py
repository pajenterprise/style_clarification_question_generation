import torch
from torch.autograd import Variable
import pdb
from .read_data import *
import numpy as np
from .constants import *


# Return a list of indexes, one for each word in the sentence, plus EOS
def prepare_sequence(seq, word2index, max_len):
    sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in seq.split(' ')[:max_len-1]]
    sequence.append(word2index[EOS_token])
    length = len(sequence)
    sequence += [word2index[PAD_token]]*(max_len - len(sequence))
    return sequence, length


def preprocess_data(triples, word2index, max_post_len, max_ques_len):
    id_seqs = []
    post_seqs = []
    post_lens = []
    ques_seqs = []
    ques_lens = []

    for i in range(len(triples)):
        curr_id, post, ques, ans = triples[i]
        id_seqs.append(curr_id)
        post_seq, post_len = prepare_sequence(post, word2index, max_post_len)
        post_seqs.append(post_seq)
        post_lens.append(post_len)
        ques_seq, ques_len = prepare_sequence(ques, word2index, max_ques_len)
        ques_seqs.append(ques_seq)
        ques_lens.append(ques_len)

    return id_seqs, post_seqs, post_lens, ques_seqs, ques_lens
