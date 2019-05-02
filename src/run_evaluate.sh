#!/bin/bash

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=D:\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src
EMB_DIR=D:\\style_clarification_question_generation\\embeddings\\$SITENAME
PARAMS_DIR=D:\\style_clarification_question_generation\\$SITENAME\\params

python $SCRIPT_DIR\\evaluate.py      --test_context $CQ_DATA_DIR\\test_context.txt \
									--test_ques $CQ_DATA_DIR\\test_ques.txt \
									--test_ids $CQ_DATA_DIR\\test_asin.txt \
									--test_pred_ques $CQ_DATA_DIR\\test_pred_ques.txt \
									--q_encoder_params $PARAMS_DIR\\q_encoder_params.epoch100 \
									--q_decoder_params $PARAMS_DIR\\q_decoder_params.epoch100 \
									--word_embeddings $EMB_DIR\\word_embeddings.p \
									--vocab $EMB_DIR\\vocab.p \
									--model seq2seq.epoch100 \
									--max_post_len 100 \
									--max_ques_len 30 \
									--batch_size 128 \
									--n_epochs 40 \
									--beam True
									#--greedy True

