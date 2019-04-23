import argparse
from nltk.corpus import stopwords
import numpy as np
import pickle as p
import scipy.sparse
from scipy import spatial
import sklearn
import string
import sys


def main(args):
    ids = [line.strip('\n') for line in open(args.ids, 'r').readlines()]
    contexts = [line.strip('\n') for line in open(args.contexts, 'r').readlines()]
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    stop_words = set(stopwords.words('english'))
    contexts_avg_emb = []
    prev_id = None
    uniq_ids = []
    translator = str.maketrans('', '', string.punctuation)
    for i, context in enumerate(contexts):
        if ids[i] == prev_id:
            continue
        else:
            prev_id = ids[i]
        uniq_ids.append(ids[i])
        context = context.translate(translator)
        context_no_stopwords = [w for w in context.split() if not w in stop_words]
        context_emb = []
        for w in context_no_stopwords:
            if w in word2index:
                context_emb.append(word_embeddings[word2index[w]])
            else:
                context_emb.append(word_embeddings[word2index['<unk>']])
        contexts_avg_emb.append(np.mean(context_emb, axis=0))
    contexts_pairwise_sim = sklearn.metrics.pairwise.cosine_similarity(contexts_avg_emb)
    contexts_pairwise_sim[contexts_pairwise_sim < 0.75] = 0
    sparse_contexts_pairwise_sim = scipy.sparse.csr_matrix(contexts_pairwise_sim)
    scipy.sparse.save_npz(args.contexts_pairwise_word_emb_sim, sparse_contexts_pairwise_sim)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ids", type = str)
    argparser.add_argument("--contexts", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--contexts_pairwise_word_emb_sim", type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)


