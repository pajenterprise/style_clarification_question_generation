import argparse
import sys
import networkx as nx
import pickle as p
from collections import OrderedDict
import operator
import scipy.sparse
from collections import defaultdict


def add_edges_between_contexts(contexts_sim, G, N):
    # Adding edges between context nodes
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if contexts_sim[i][j] > 0.9:
                wt = contexts_sim[i][j]
                #print('%d, %d, %.4f' % (i, j, wt))
                G.add_edge(i, j, weight=wt)
                G.add_edge(j, i, weight=wt)


def add_unique_question_nodes(questions_sim, question_node_map, G, N, start, end):
    M = len(questions_sim)
    for j in range(end - start):
        create_new_node = True
        max_sim = 0.95
        max_sim_i = -1
        for i in range(M):
            if i == j:
                continue
            if questions_sim[i][j] > max_sim:
                create_new_node = False
                max_sim = questions_sim[i][j]
                max_sim_i = i
        if not create_new_node:
            if G.has_node(N+max_sim_i):
                question_node_map[N+start+j] = N+max_sim_i
            elif N+max_sim_i in question_node_map:
                question_node_map[N+start+j] = question_node_map[N+max_sim_i]
            else:
                G.add_node(N+start+j)
        else:
            G.add_node(N+start+j)


def add_edges_between_questions(questions_sim, question_node_map, G, N, start, end):
    M = len(questions_sim)
    for j in range(end-start):
        for i in range(M):
            if questions_sim[i][j] > 0.9:
                wt = questions_sim[i][j]
                if G.has_node(N + i):
                    src = N + i
                else:
                    src = question_node_map[N+i]
                if G.has_node(N + j + start):
                    tgt = N + j + start
                elif (N + j + start) in question_node_map:
                    tgt = question_node_map[N + j + start]
                else:
                    print('Not a node and does not map to current node! ')
                    tgt = N + j + start
                G.add_edge(src, tgt, weight=wt)
                G.add_edge(tgt, src, weight=wt)


def add_edges_between_context_question(ids, question_node_map, G, N):
    prev_id = -1
    uniq_id_ix = -1
    # Adding edges between context and question nodes
    for i in range(len(ids)):
        if ids[i] != prev_id:
            uniq_id_ix += 1
            prev_id = ids[i]
        src = uniq_id_ix
        if G.has_node(N+i):
            tgt = N+i
        else:
            tgt = question_node_map[N+i]
        G.add_edge(src, tgt, weight=1.0)
        #G.add_edge(tgt, src, weight=0.5)


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

    question_node_map = {}

    print('Adding edges between contexts')
    add_edges_between_contexts(contexts_sim, G, N)
    questions_sim_parts = []
    print('Loading question sim')
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part1).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part2).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part3).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part4).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part5).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part6).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part7).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part8).toarray())
    questions_sim_parts.append(scipy.sparse.load_npz(args.questions_sim_part9).toarray())
    print('Adding question nodes')
    for part in range(9):
        start = part*10000
        if part == 8:
            end = len(ids)
        else:
            end = (part+1)*10000
        add_unique_question_nodes(questions_sim_parts[part], question_node_map, G, N, start=start, end=end)
    print('Adding edges between questions')
    for part in range(9):
        start = part * 10000
        if part == 8:
            end = len(ids)
        else:
            end = (part + 1) * 10000
        add_edges_between_questions(questions_sim_parts[part], question_node_map, G, N, start=start, end=end)

    print('Adding edges between contexts and questions')
    add_edges_between_context_question(ids, question_node_map, G, N)

    print('Running pagerank algorithm')
    pagerank = nx.pagerank(G)
    ques_wt_dict = {}
    for i in range(len(ids)):
        if G.has_node(N+i):
            ques_wt_dict[ids[i]+'_'+str(i)] = pagerank[N+i]
        else:
            ques_wt_dict[ids[i]+'_'+str(i)] = pagerank[question_node_map[N+i]]

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
    argparser.add_argument("--outfile", type=str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
    
