#!/bin/bash

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=D:\\style_clarification_question_generation\\$SITENAME\\relevance_classifier_data
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src\\relevance_classifier
EMB_DIR=D:\\style_clarification_question_generation\\embeddings\\$SITENAME
PARAMS_DIR=D:\\style_clarification_question_generation\\$SITENAME\\params

python $SCRIPT_DIR\\relevance_classifier.py         --contexts $CQ_DATA_DIR\\train_contexts_relevance_classifier.txt \
                                                    --questions $CQ_DATA_DIR\\train_ques_relevance_classifier.txt \
                                                    --ids $CQ_DATA_DIR\\train_asin_relevance_classifier.txt \
                                                    --labels $CQ_DATA_DIR\\train_labels_relevance_classifier.txt \
                                                    --relevance_classifier_params $PARAMS_DIR\\relevance_classifier_params \
                                                    --context_params $PARAMS_DIR\\context_params \
                                                    --question_params $PARAMS_DIR\\question_params \
                                                    --word_embeddings $EMB_DIR\\word_embeddings.p \
                                                    --vocab $EMB_DIR\\vocab.p \
                                                    --n_epochs 50 \
                                                    --max_post_len 100 \
                                                    --max_ques_len 20 \






