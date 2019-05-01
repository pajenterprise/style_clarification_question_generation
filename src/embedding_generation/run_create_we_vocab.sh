#!/bin/bash

SITENAME=Home_and_Kitchen

SCRIPTS_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src\\embedding_generation
EMB_DIR=D:\\style_clarification_question_generation\\embeddings\\$SITENAME

python $SCRIPTS_DIR\\create_we_vocab.py $EMB_DIR\\vectors.txt $EMB_DIR\\word_embeddings.p $EMB_DIR\\vocab.p

