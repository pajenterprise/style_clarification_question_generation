#!/bin/bash

SITENAME=Home_and_Kitchen
STYLE_CQ_DATA_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\$SITENAME
SCRIPT_DIR=C:\\Users\\sudhra\\style_clarification_question_generation\\src
EMB_DIR=C:\\Users\\sudhra\\clarification_question_generation_pytorch\\embeddings\\$SITENAME

python $SCRIPT_DIR\\kmeans_clustering_questions.py	--word_embeddings $EMB_DIR\\word_embeddings.p \
														--vocab $EMB_DIR\\vocab.p \
														--questions $STYLE_CQ_DATA_DIR\\train_ques.txt \
                                                        --ids $STYLE_CQ_DATA_DIR\\train_asin.txt \


