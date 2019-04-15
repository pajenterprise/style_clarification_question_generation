import argparse
import sys
import networkx as nx
import pickle as p
from collections import OrderedDict
import operator
import scipy.sparse


def add_edges_between_contexts(contexts_sim, G, N):
    # Adding edges between context nodes
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if contexts_sim[i][j] > 0.8:
                wt = contexts_sim[i][j]
                #print('%d, %d, %.4f' % (i, j, wt))
                G.add_edge(i, j, weight=wt)
                G.add_edge(j, i, weight=wt)


def add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start, end):
    # Adding edges between question nodes
    for i in range(M):
        for j in range(end-start):
            if i == j:
                continue
            if questions_sim[i][j] > 0.8:
                wt = questions_sim[i][j]
                #print('%d, %d, %.4f' % (i, j, wt))
                #print('%s %s %.4f' % (ques_dict[ids[i]+'_'+str(i)], ques_dict[ids[j]+'_'+str(j)], wt))
                G.add_edge(N+i, N+j+start, weight=wt)
                G.add_edge(N+j+start, N+i, weight=wt)


def add_edges_between_context_question(ids, G, N):
    prev_id = -1
    uniq_id_ix = -1
    # Adding edges between context and question nodes
    for i in range(len(ids)):
        if ids[i] != prev_id:
            uniq_id_ix += 1
            prev_id = ids[i]
        G.add_edge(uniq_id_ix, N + i, weight=0.5)
        G.add_edge(N + i, uniq_id_ix, weight=0.5)


def main(args):
    print('Loading contexts sim ')
    sparse_contexts_sim = scipy.sparse.load_npz(args.contexts_sim)
    contexts_sim = sparse_contexts_sim.toarray()
    ids = [line.strip('\n') for line in open(args.ids, 'r').readlines()]
    i = 0
    ques_dict = {}
    print('Reading questions')
    for line in open(args.questions, 'r').readlines():
        ques = line.strip('\n')
        ques_dict[ids[i]+'_'+str(i)] = ques
        i += 1
    
    G = nx.DiGraph()
    N = len(contexts_sim)
    print('Adding context nodes')
    G.add_nodes_from(range(N))
    M = len(ids)
    print('Adding question nodes')
    G.add_nodes_from(range(N, N+M))
    print('Adding edges between contexts')
    add_edges_between_contexts(contexts_sim, G, N)
    contexts_sim = None    

    print('Loading question sim part1')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part1).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=0, end=10000)

    print('Loading question sim part2')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part2).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=10000, end=20000)

    print('Loading question sim part3')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part3).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=20000, end=30000)

    print('Loading question sim part4')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part4).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=30000, end=40000)

    print('Loading question sim part5')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part5).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=40000, end=50000)

    print('Loading question sim part6')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part6).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=50000, end=60000)

    print('Loading question sim part7')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part7).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=60000, end=70000)

    print('Loading question sim part8')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part8).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=70000, end=80000)

    print('Loading question sim part9')
    questions_sim = scipy.sparse.load_npz(args.questions_sim_part9).toarray()
    print('Adding edges between questions')
    add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=80000, end=len(ids))

    print('Adding edges between contexts and questions')
    add_edges_between_context_question(ids, G, N)

    print('Running pagerank algorithm')
    pagerank = nx.pagerank(G)
    ques_wt_dict = {}
    for i in range(len(ids)):
        ques_wt_dict[ids[i]+'_'+str(i)] = pagerank[i]
    sorted_x = sorted(ques_wt_dict.items(), key=operator.itemgetter(1))

    print('Writing to outfile')
    outfile = open(args.outfile, 'w')
    for curr_id, wt in sorted_x:
        outfile.write(curr_id+'\n')
        outfile.write(ques_dict[curr_id]+'\n')
        outfile.write('%.6f\n' % wt)
        outfile.write('\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ids", type = str)
    argparser.add_argument("--questions", type = str)
    argparser.add_argument("--contexts_sim", type = str)
    argparser.add_argument("--questions_sim_part1", type = str)
    argparser.add_argument("--questions_sim_part2", type = str)
    argparser.add_argument("--questions_sim_part3", type = str)
    argparser.add_argument("--questions_sim_part4", type = str)
    argparser.add_argument("--questions_sim_part5", type = str)
    argparser.add_argument("--questions_sim_part6", type = str)
    argparser.add_argument("--questions_sim_part7", type = str)
    argparser.add_argument("--questions_sim_part8", type = str)
    argparser.add_argument("--questions_sim_part9", type = str)
    argparser.add_argument("--outfile", type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
    
