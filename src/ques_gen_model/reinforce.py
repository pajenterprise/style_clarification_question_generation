import argparse
import pickle as p
import sys

import torch
import torch.nn as nn
from torch import optim
import time

from ques_gen_model.encoderRNN import *
from ques_gen_model.attnDecoderRNN import *
from ques_gen_model.read_data import *
from ques_gen_model.prepare_data import *
from ques_gen_model.baselineFF import *
from relevance_classifier.RNN import *
from relevance_classifier.FeedForward import *
from ques_gen_model.reinforce_train import *
from ques_gen_model.helper import *
from ques_gen_model.constants import *


def run_reinforce(train_data, test_data, word_embeddings, word2index, index2word, args):
    tr_id_seqs, tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens = train_data

    te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens = test_data

    print('Defining encoder decoder models')
    q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
    q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

    if USE_CUDA:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        q_encoder = q_encoder.to(device)
        q_decoder = q_decoder.to(device)

    # Load encoder, decoder params
    print('Loading encoded, decoder params')
    q_encoder.load_state_dict(torch.load(args.q_encoder_params))
    q_decoder.load_state_dict(torch.load(args.q_decoder_params))

    q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE)
    q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

    context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=2)
    question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=2)
    relevance_model = FeedForward(HIDDEN_SIZE*2*2)

    baseline_model = BaselineFF(HIDDEN_SIZE)

    if USE_CUDA:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        context_model.to(device)
        question_model.to(device)
        relevance_model.to(device)
        baseline_model.to(device)

    # Load utility calculator model params
    print('Loading utility model params')
    context_model.load_state_dict(torch.load(args.context_params))
    question_model.load_state_dict(torch.load(args.question_params))
    relevance_model.load_state_dict(torch.load(args.relevance_classifier_params))

    for param in context_model.parameters(): param.requires_grad = False
    for param in question_model.parameters(): param.requires_grad = False
    for param in relevance_model.parameters(): param.requires_grad = False

    baseline_optimizer = optim.Adam(baseline_model.parameters())
    baseline_criterion = torch.nn.MSELoss()

    epoch = 0.
    start = time.time()
    n_batches = len(tr_post_seqs)/args.batch_size
    mixer_delta = args.max_ques_len
    while epoch < args.n_epochs:
        epoch += 1
        total_loss = 0.
        total_xe_loss = 0.
        total_rl_loss = 0.
        total_u_pred = 0.
        total_u_b_pred = 0.
        if mixer_delta >= 2:
            mixer_delta = mixer_delta - 2
        batch_num = 0
        for ids, post, pl, ques, ql in iterate_minibatches(tr_id_seqs, tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                                                      args.batch_size):
            batch_num += 1
            xe_loss, rl_loss, reward, b_reward = reinforce_train(post, pl, ques, ql, q_encoder, q_decoder,
                                                                 q_encoder_optimizer, q_decoder_optimizer,
                                                                 baseline_model, baseline_optimizer, baseline_criterion,
                                                                 context_model, question_model, relevance_model,
                                                                 word2index, index2word, mixer_delta, args)
            total_u_pred += reward.data.sum() / args.batch_size
            total_u_b_pred += b_reward.data.sum() / args.batch_size
            total_xe_loss += xe_loss
            total_rl_loss += rl_loss

        print_loss_avg = total_loss / n_batches
        print_xe_loss_avg = total_xe_loss / n_batches
        print_rl_loss_avg = total_rl_loss / n_batches
        print_u_pred_avg = total_u_pred / n_batches
        print_u_b_pred_avg = total_u_b_pred / n_batches
        print_summary = '%s %d Loss: %.4f XE_loss: %.4f RL_loss: %.4f U_pred: %.4f B_pred: %.4f' % \
                        (time_since(start, epoch / args.n_epochs), epoch, print_loss_avg, print_xe_loss_avg,
                         print_rl_loss_avg, print_u_pred_avg, print_u_b_pred_avg)
        print(print_summary)

        print('Saving RL model params')
        torch.save(q_encoder.state_dict(), args.q_encoder_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(q_decoder.state_dict(), args.q_decoder_params + '.' + args.model + '.epoch%d' % epoch)

