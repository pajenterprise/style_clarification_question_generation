#!/bin/bash

#SBATCH --job-name=pagerank_train_word_emb_sim_part1
#SBATCH --output=pagerank_train_word_emb_sim_part1
#SBATCH --qos=batch
#SBATCH --mem=128g
#SBATCH --time=2:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
CQ_DATA_DIR=/fs/clip-amr/style_clarification_question_generation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/style_clarification_question_generation/src
SCRATCH_CQ_DATA_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/pagerank.py  --ids $CQ_DATA_DIR/train_asin.txt \
                                --questions $CQ_DATA_DIR/train_ques.txt \
                                --contexts_sim $SCRATCH_CQ_DATA_DIR/train_context_pairwise_word_emb_sim.threshold0.5.npz \
                                --questions_sim_part1 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.threshold0.5.part1.npz \
                                --questions_sim_part2 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part2.npz \
                                --questions_sim_part3 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part3.npz \
                                --questions_sim_part4 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part4.npz \
                                --questions_sim_part5 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part5.npz \
                                --questions_sim_part6 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part6.npz \
                                --questions_sim_part7 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part7.npz \
                                --questions_sim_part8 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part8.npz \
                                --questions_sim_part9 $SCRATCH_CQ_DATA_DIR/train_question_pairwise_word_emb_sim.part9.npz \
								--outfile $SCRATCH_CQ_DATA_DIR/train_pairwise_word_emb_sim_pagerank.part1.out \

