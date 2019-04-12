#!/bin/bash

#SBATCH --job-name=train_context_pairwise_word_emb_sim
#SBATCH --output=train_context_pairwise_word_emb_sim
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
CQ_DATA_DIR=/fs/clip-amr/style_clarification_question_generation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/style_clarification_question_generation/src
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME/200
SCRATCH_CQ_DATA_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/compute_context_word_emb_sim.py	--word_embeddings $EMB_DIR/word_embeddings.p \
													--vocab $EMB_DIR/vocab.p \
													--ids $CQ_DATA_DIR/train_asin.txt \
													--contexts $CQ_DATA_DIR/train_context.txt \
													--contexts_pairwise_word_emb_sim $SCRATCH_CQ_DATA_DIR/train_context_pairwise_word_emb_sim.threshold0.5.npz \

