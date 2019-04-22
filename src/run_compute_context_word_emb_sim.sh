#!/bin/bash

SITENAME=Home_and_Kitchen
STYLE_CQ_DATA_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src
EMB_DIR=C:\\Users\\sudhra\\clarification_question_generation_pytorch\\embeddings\\$SITENAME

python $SCRIPT_DIR\\compute_context_word_emb_sim.py     --word_embeddings $EMB_DIR\\word_embeddings.p \
													    --vocab $EMB_DIR\\vocab.p \
													    --ids $STYLE_CQ_DATA_DIR\\train_asin.txt \
													    --contexts $STYLE_CQ_DATA_DIR\\train_context.txt \
													    --contexts_pairwise_word_emb_sim $STYLE_CQ_DATA_DIR\\train_context_pairwise_word_emb_sim.threshold0.75.npz \

