import argparse
import pickle as p
from nltk.corpus import stopwords
import numpy as np
import scipy.sparse
import string
from scipy import spatial
import sklearn
import sys

def main(args):
    ids = [line.strip('\n') for line in open(args.ids, 'r').readlines()]
    questions = [line.strip('\n') for line in open(args.questions, 'r').readlines()]
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    stop_words = set(stopwords.words('english'))
    questions_avg_emb = []
    print('Computing avg emb...')
    translator = str.maketrans('', '', string.punctuation)
    for i, question in enumerate(questions):
        org_question = question
        question = question.translate(translator)
        question_no_stopwords = [w for w in question.split() if not w in stop_words]
        if len(question_no_stopwords) == 0:
            question_no_stopwords = question.split()
        if question.strip() == '':
            question_no_stopwords = org_question
        question_emb = []
        for w in question_no_stopwords:
            if w in word2index:
                question_emb.append(word_embeddings[word2index[w]])
            else:
                question_emb.append(word_embeddings[word2index['<unk>']])
        questions_avg_emb.append(np.mean(question_emb, axis=0))
    print('Done')
    print(len(questions_avg_emb))
    start = int(args.start)
    if args.end:
        end = int(args.end)
    else:
        end = len(questions_avg_emb)
    questions_pairwise_sim = sklearn.metrics.pairwise.cosine_similarity(questions_avg_emb, questions_avg_emb[start:end])
    questions_pairwise_sim[questions_pairwise_sim < 0.75] = 0
    questions_pairwise_sim = scipy.sparse.csr_matrix(questions_pairwise_sim)
    scipy.sparse.save_npz(args.questions_pairwise_word_emb_sim, questions_pairwise_sim)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ids", type = str)
    argparser.add_argument("--questions", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--start", type = str)
    argparser.add_argument("--end", type = str)
    argparser.add_argument("--questions_pairwise_word_emb_sim", type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)

