#!/bin/bash

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=D:\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src
EMB_DIR=D:\\style_clarification_question_generation\\embeddings\\$SITENAME
PARAMS_DIR=D:\\style_clarification_question_generation\\$SITENAME\\params

python $SCRIPT_DIR\\read_model_outputs.py         --test_context $CQ_DATA_DIR\\test_context_with_labels.txt \
                                    --test_ques $CQ_DATA_DIR\\test_ques.txt \
									--test_ids $CQ_DATA_DIR\\test_asin.txt \
                                    --pred_test_question_ids $CQ_DATA_DIR\\test_pred_ques.txt.seq2seq.epoch100.beam0.ids \
                                    --pred_test_question_seq2seq $CQ_DATA_DIR\\test_pred_ques.txt.seq2seq.epoch100.beam0 \
                                    --pred_test_question_togeneric $CQ_DATA_DIR\\test_pred_ques.txt.seq2seq_togeneric.epoch100.beam0 \
                                    --pred_test_question_tospecific $CQ_DATA_DIR\\test_pred_ques.txt.seq2seq_tospecific.epoch100.beam0 \
                                    --analysis_output_file $CQ_DATA_DIR\\compare_model_outputs.csv