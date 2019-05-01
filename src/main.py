import argparse
import os
import pickle as p
import string
import time
import datetime
import math
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

import numpy as np

from seq2seq.read_data import *
from seq2seq.prepare_data import *
from seq2seq.masked_cross_entropy import *
from seq2seq.seq2seq import *
from seq2seq.constants import *


def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))

    train_data = read_data(args.train_context, args.train_question, args.train_ids,
                            args.max_post_len, args.max_ques_len)
    if args.tune_ids is not None:
        test_data = read_data(args.tune_context, args.tune_question, args.tune_ids,
                              args.max_post_len, args.max_ques_len)
    else:
        test_data = read_data(args.tune_context, args.tune_question, None,
                              args.max_post_len, args.max_ques_len)

    print('No. of train_data %d' % len(train_data))
    print('No. of test_data %d' % len(test_data))

    ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens = \
        preprocess_data(train_data, word2index, args.max_post_len, args.max_ques_len)

    q_train_data = ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens

    ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens = \
        preprocess_data(test_data, word2index, args.max_post_len, args.max_ques_len)

    q_test_data = ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens

    run_seq2seq(q_train_data, q_test_data, word2index, word_embeddings,
                    args.q_encoder_params, args.q_decoder_params,
                    args.max_ques_len, args.n_epochs, args.batch_size, n_layers=2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_context", type = str)
    argparser.add_argument("--train_question", type = str)
    argparser.add_argument("--train_ids", type=str)
    argparser.add_argument("--tune_context", type = str)
    argparser.add_argument("--tune_question", type = str)
    argparser.add_argument("--tune_ids", type=str)
    argparser.add_argument("--test_context", type = str)
    argparser.add_argument("--test_question", type = str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--q_encoder_params", type = str)
    argparser.add_argument("--q_decoder_params", type = str)
    argparser.add_argument("--context_params", type = str)
    argparser.add_argument("--question_params", type = str)
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--max_post_len", type = int, default=300)
    argparser.add_argument("--max_ques_len", type = int, default=50)
    argparser.add_argument("--n_epochs", type = int, default=20)
    argparser.add_argument("--batch_size", type = int, default=128)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)

