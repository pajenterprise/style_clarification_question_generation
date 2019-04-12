#!/bin/bash

#SBATCH --job-name=train_question_pairwise_word_emb_sim_part2
#SBATCH --output=train_question_pairwise_word_emb_sim_part2
#SBATCH --qos=batch
#SBATCH --mem=128g
#SBATCH --time=1:00:00

SITENAME=Home_and_Kitchen
CQ_DATA_DIR=/fs/clip-amr/style_clarification_question_generation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/style_clarification_question_generation/src
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME/200
SCRATCH_CQ_DATA_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/compute_question_word_emb_sim.py	--word_embeddings $EMB_DIR/word_embeddings.p \
														--vocab $EMB_DIR/vocab.p \
														--ids $CQ_DATA_DIR/train_asin.txt \
														--questions $CQ_DATA_DIR/train_ques.txt \
														--start 10000 \
														--end 20000 \
														--questions_pairwise_word_emb_sim $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.threshold0.5.part2.npz \

