import argparse
import csv
import networkx as nx
import numpy as np
import random
import scipy
import sys


def add_edges_between_contexts(contexts_sim, G, N):
    # Adding edges between context nodes
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if contexts_sim[i][j] > 0.9:
                G.add_edge(i, j)


def add_edges_between_questions(questions_sim, G, N, start, end):
    # M = len(questions_sim[0])
    M = len(questions_sim)
    for j in range(end - start):
        for i in range(M):
            if i == j:
                continue
            if questions_sim[i][j] > 0.9:
                G.add_edge(N+i, N+j+start)


def add_edges_between_context_question(ids, G, N):
    prev_id = -1
    uniq_id_ix = -1
    # Adding edges between context and question nodes
    for i in range(len(ids)):
        if ids[i] != prev_id:
            uniq_id_ix += 1
            prev_id = ids[i]
        src = uniq_id_ix
        tgt = N + i
        G.add_edge(src, tgt)


def main(args):
    print('Loading contexts sim ')
    sparse_contexts_sim = scipy.sparse.load_npz(args.contexts_sim)
    contexts_sim = sparse_contexts_sim.toarray()
    ids = [line.strip('\n') for line in open(args.ids, 'r').readlines()]
    i = 0
    ques_dict = {}
    questions = []
    print('Reading questions')
    for line in open(args.questions, 'r').readlines():
        ques = line.strip('\n')
        questions.append(ques)
        ques_dict[ids[i] + '_' + str(i)] = ques
        i += 1
    contexts = [line.strip('\n') for line in open(args.contexts, 'r').readlines()]
    G = nx.Graph()
    N = len(contexts_sim)
    print('Adding context %d nodes' % N)
    G.add_nodes_from(range(N))

    print('Adding edges between contexts')
    add_edges_between_contexts(contexts_sim, G, N)
    print('Loading question sim')
    questions_sim_parts = []
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part1).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part2).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part3).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part4).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part5).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part6).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part7).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part8).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part9).toarray())
    print('Adding %d question nodes' % (len(questions_sim_parts[0])))
    G.add_nodes_from(range(N, N+len(questions_sim_parts[0])))
    print('Adding edges between questions')
    for part in range(9):
        start = part * 10000
        if part == 8:
            end = len(ids)
        else:
            end = (part + 1) * 10000
        add_edges_between_questions(questions_sim_parts[part], G, N, start=start, end=end)

    print('Adding edges between contexts and questions')
    add_edges_between_context_question(ids, G, N)

    print('Writing to outfile')
    ids_out_file = open(args.ids_out, 'w')
    contexts_out_file = open(args.contexts_out, 'w')
    questions_out_file = open(args.questions_out, 'w')
    labels_out_file = open(args.labels_out, 'w')
    for i in range(len(ids)):
        print(ids[i])
        ids_out_file.write('%s\n' % ids[i])
        ids_out_file.write('%s\n' % ids[i])
        contexts_out_file.write('%s\n' % contexts[i])
        contexts_out_file.write('%s\n' % contexts[i])
        questions_out_file.write('%s\n' % questions[i])
        rand = random.randint(0, len(ids)-1)
        while nx.has_path(G, N+i, N+rand) or i == rand:
            rand = random.randint(0, len(ids)-1)
        irrelevant_question = questions[rand]
        questions_out_file.write('%s\n' % irrelevant_question)
        labels_out_file.write('1\n')
        labels_out_file.write('0\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ids", type=str)
    argparser.add_argument("--contexts", type=str)
    argparser.add_argument("--questions", type=str)
    argparser.add_argument("--contexts_sim", type=str)
    argparser.add_argument("--questions_sim_part1", type=str)
    argparser.add_argument("--questions_sim_part2", type=str)
    argparser.add_argument("--questions_sim_part3", type=str)
    argparser.add_argument("--questions_sim_part4", type=str)
    argparser.add_argument("--questions_sim_part5", type=str)
    argparser.add_argument("--questions_sim_part6", type=str)
    argparser.add_argument("--questions_sim_part7", type=str)
    argparser.add_argument("--questions_sim_part8", type=str)
    argparser.add_argument("--questions_sim_part9", type=str)
    argparser.add_argument("--ids_out", type=str)
    argparser.add_argument("--contexts_out", type=str)
    argparser.add_argument("--questions_out", type=str)
    argparser.add_argument("--labels_out", type=str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)

