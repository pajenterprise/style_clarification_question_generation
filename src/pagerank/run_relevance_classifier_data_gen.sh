#!/bin/bash

SITENAME=Home_and_Kitchen
STYLE_CQ_DATA_DIR=D:\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src\\pagerank

python $SCRIPT_DIR\\relevance_classifier_data_gen.py    --ids $STYLE_CQ_DATA_DIR\\train_asin.txt \
                                                        --questions $STYLE_CQ_DATA_DIR\\train_ques.txt \
                                                        --contexts $STYLE_CQ_DATA_DIR\\train_context.txt \
                                                        --contexts_sim $STYLE_CQ_DATA_DIR\\train_context_pairwise_word_emb_sim.threshold0.75.npz \
                                                        --questions_sim_part1 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part1.npz \
                                                        --questions_sim_part2 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part2.npz \
                                                        --questions_sim_part3 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part3.npz \
                                                        --questions_sim_part4 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part4.npz \
                                                        --questions_sim_part5 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part5.npz \
                                                        --questions_sim_part6 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part6.npz \
                                                        --questions_sim_part7 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part7.npz \
                                                        --questions_sim_part8 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part8.npz \
                                                        --questions_sim_part9 $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part9.npz \
                                                        --ids_out $STYLE_CQ_DATA_DIR\\train_asin_relevance_classifier.txt \
                                                        --questions_out $STYLE_CQ_DATA_DIR\\train_ques_relevance_classifier.txt \
                                                        --contexts_out $STYLE_CQ_DATA_DIR\\train_contexts_relevance_classifier.txt \
                                                        --labels_out $STYLE_CQ_DATA_DIR\\train_labels_relevance_classifier.txt \
