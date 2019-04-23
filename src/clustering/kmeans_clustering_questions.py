import argparse
import pickle as p
from nltk.corpus import stopwords
import numpy as np
import scipy.sparse
import string
from scipy import spatial
import sklearn
import sys
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

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
    clustering = AgglomerativeClustering(n_clusters=int(len(questions)/2), affinity="cosine", linkage="average")
    clustering.fit(questions_avg_emb)
    question_clusters = defaultdict(list)
    for i, question in enumerate(questions):
        question_clusters[clustering.labels_[i]].append(i)
    for label, cluster in question_clusters.items():
        cluster_avg_emb = []
        for i in cluster:
            print('%d: %s: %s' % (label, ids[i], questions[i]))
            cluster_avg_emb.append(questions_avg_emb[i])
        questions_pairwise_sim = sklearn.metrics.pairwise.cosine_similarity(cluster_avg_emb)
        questions_pairwise_sim[questions_pairwise_sim < 0.9] = 0
        print(len(questions_pairwise_sim))
        if np.count_nonzero(questions_pairwise_sim) < 0.1*len(questions_pairwise_sim)*len(questions_pairwise_sim):
            print('BAD CLUSTER')
        else:
            print('GOOD CLUSTER')
        print('\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ids", type=str)
    argparser.add_argument("--questions", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--questions_pairwise_word_emb_sim", type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)

