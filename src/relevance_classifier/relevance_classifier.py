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
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

import numpy as np

from read_data import *
from prepare_data import *
from RNN import *
from FeedForward import *
from train import *
from evaluate import *
from masked_cross_entropy import *
from constants import *


def run_classifier(train_data, test_data, word_embeddings, args, n_layers):
    context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers)
    question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers)
    relevance_model = FeedForward(HIDDEN_SIZE*2*2)

    if USE_CUDA:
        word_embeddings = Variable(torch.FloatTensor(word_embeddings).cuda())
    else:
        word_embeddings = Variable(torch.FloatTensor(word_embeddings))

    context_model.embedding.weight.data.copy_(word_embeddings)
    question_model.embedding.weight.data.copy_(word_embeddings)

    # Fix word embeddings
    context_model.embedding.weight.requires_grad = True
    question_model.embedding.weight.requires_grad = True

    optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) +
                            list([par for par in question_model.parameters() if par.requires_grad]) +
                            list([par for par in relevance_model.parameters() if par.requires_grad]))

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    if USE_CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        context_model = context_model.to(device)
        question_model = question_model.to(device)
        relevance_model = relevance_model.to(device)
        criterion = criterion.to(device)

    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_loss, train_acc = train_fn(context_model, question_model, relevance_model,
                                         train_data, optimizer, criterion, args)
        valid_loss, valid_acc = evaluate(context_model, question_model, relevance_model,
                                         test_data, criterion, args)
        print('Epoch %d: Train Loss: %.3f, Train Acc: %.3f, Val Loss: %.3f, Val Acc: %.3f' % (epoch, train_loss, train_acc, valid_loss, valid_acc))
        print('Time taken: ', time.time()-start_time)
        if epoch % 5 == 0:
            print('Saving model params')
            torch.save(context_model.state_dict(), args.context_params+'.epoch%d' % epoch)
            torch.save(question_model.state_dict(), args.question_params+'.epoch%d' % epoch)
            torch.save(relevance_model.state_dict(), args.relevance_classifier_params+'.epoch%d' % epoch)


def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))

    data = read_data(args.contexts, args.questions, args.ids, args.labels,
                            args.max_post_len, args.max_ques_len)

    N = len(data)
    train_data = data[:int(0.8*N)]
    test_data = data[int(0.8*N):]

    print('No. of train_data %d' % len(train_data))
    print('No. of test_data %d' % len(test_data))

    ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens, lab_seqs = \
        preprocess_data(train_data, word2index, args.max_post_len, args.max_ques_len)

    q_train_data = ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens, lab_seqs

    ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens, lab_seqs = \
        preprocess_data(test_data, word2index, args.max_post_len, args.max_ques_len)

    q_test_data = ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens, lab_seqs

    run_classifier(q_train_data, q_test_data, word_embeddings, args, n_layers=2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ids", type=str)
    argparser.add_argument("--contexts", type = str)
    argparser.add_argument("--questions", type = str)
    argparser.add_argument("--labels", type=str)
    argparser.add_argument("--context_params", type=str)
    argparser.add_argument("--question_params", type=str)
    argparser.add_argument("--relevance_classifier_params", type = str)
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

