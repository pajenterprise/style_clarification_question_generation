#!/bin/bash

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=D:\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src
EMB_DIR=D:\\style_clarification_question_generation\\embeddings\\$SITENAME
PARAMS_DIR=D:\\style_clarification_question_generation\\$SITENAME\\params

python $SCRIPT_DIR\\reinforce_model.py      --train_context $CQ_DATA_DIR\\train_context_with_labels.txt \
                                            --train_ques $CQ_DATA_DIR\\train_ques.txt \
                                            --train_ids $CQ_DATA_DIR\\train_asin.txt \
                                            --tune_context $CQ_DATA_DIR\\tune_context_with_labels.txt \
                                            --tune_ques $CQ_DATA_DIR\\tune_ques.txt \
                                            --tune_ids $CQ_DATA_DIR\\tune_asin.txt \
                                            --test_context $CQ_DATA_DIR\\test_context_with_labels.txt \
                                            --test_ques $CQ_DATA_DIR\\test_ques.txt \
                                            --test_ids $CQ_DATA_DIR\\test_asin.txt \
                                            --q_encoder_params $PARAMS_DIR\\q_encoder_params_genspec.epoch100 \
                                            --q_decoder_params $PARAMS_DIR\\q_decoder_params_genspec.epoch100 \
                                            --context_params $PARAMS_DIR\\context_params.epoch10 \
                                            --question_params $PARAMS_DIR\\question_params.epoch10 \
                                            --relevance_classifier_params $PARAMS_DIR\\relevance_classifier_params.epoch10 \
                                            --word_embeddings $EMB_DIR\\word_embeddings.p \
                                            --vocab $EMB_DIR\\vocab.p \
                                            --n_epochs 100 \
                                            --max_post_len 100 \
                                            --max_ques_len 20 \