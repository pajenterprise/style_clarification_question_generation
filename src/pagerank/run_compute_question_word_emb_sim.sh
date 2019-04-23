#!/bin/bash

SITENAME=Home_and_Kitchen
STYLE_CQ_DATA_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src
EMB_DIR=C:\\Users\\sudhra\\clarification_question_generation_pytorch\\embeddings\\$SITENAME

python $SCRIPT_DIR\\compute_question_word_emb_sim.py	--word_embeddings $EMB_DIR\\word_embeddings.p \
														--vocab $EMB_DIR\\vocab.p \
														--ids $STYLE_CQ_DATA_DIR\\train_asin.txt \
														--questions $STYLE_CQ_DATA_DIR\\train_ques.txt \
														--questions_pairwise_word_emb_sim $STYLE_CQ_DATA_DIR\\train_question_pairwise_word_emb_sim.threshold0.75.part1.npz \
														--start 0 \
														--end 10000 \


