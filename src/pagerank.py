import argparse
import sys
import networkx as nx
import cPickle as p
from collections import OrderedDict
import operator
import scipy.sparse


def add_edges_between_contexts(contexts_sim, G, N):
	# Adding edges between context nodes
	for i in range(N):
		for j in range(N):
			if i == j:
				continue
			if contexts_sim[i][j] > 0.5:
				wt = contexts_sim[i][j]
				G.add_edge(i, j, weight=wt)
				G.add_edge(j, i, weight=wt)


def add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start, end):
	# Adding edges between question nodes
	for i in range(M):
		for j in range(start, end):
			if i == j:
				continue
			if questions_sim[i][j] > 0.5:
				wt = questions_sim[i][j]
				#print ques_dict[ids[i]+'_'+str(i)], ques_dict[ids[j]+'_'+str(j)], wt
				G.add_edge(N+i, N+j, weight=wt)
				G.add_edge(N+j, N+i, weight=wt)


def add_edges_between_context_question(ids, G, N):
	prev_id = -1
	uniq_id_ix = -1
	# Adding edges between context and question nodes
	for i in range(len(ids)):
		if ids[i] != prev_id:
			uniq_id_ix += 1
			prev_id = ids[i]
			G.add_edge(uniq_id_ix, N+i, weight=1)
		else:
			G.add_edge(uniq_id_ix, N+i, weight=1)


def main(args):
	sparse_contexts_sim = scipy.sparse.load_npz(args.contexts_sim)
	contexts_sim = sparse_contexts_sim.toarray()
	ids = [line.strip('\n') for line in open(args.ids, 'r').readlines()]
	i = 0
	ques_dict = {}
	for line in open(args.questions, 'r').readlines():
		ques = line.strip('\n')
		ques_dict[ids[i]+'_'+str(i)] = ques
		i += 1
	
	G = nx.DiGraph()
	N = len(contexts_sim)
	G.add_nodes_from(range(N))
	M = len(ids)
	G.add_nodes_from(range(N, N+M))
	add_edges_between_contexts(contexts_sim, G, N)
	contexts_sim = None	

	questions_sim = scipy.sparse.load_npz(args.questions_sim_part1).toarray()
	add_edges_between_questions(questions_sim, ques_dict, ids, G, N, M, start=0, end=10000)
	questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part2).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=10000, end=20000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part3).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=20000, end=30000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part4).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=30000, end=40000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part5).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=40000, end=50000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part6).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=50000, end=60000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part7).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=60000, end=70000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part8).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=70000, end=80000)
	#questions_sim = None

	#questions_sim = scipy.sparse.load_npz(args.questions_sim_part9).toarray()
	#add_edges_between_questions(questions_sim, G, M, start=80000, end=len(ids))
	#questions_sim = None

	add_edges_between_context_question(ids, G, N)

	pagerank = nx.pagerank(G)
	ques_wt_dict = {}
	for i in range(len(ids)):
		ques_wt_dict[ids[i]+'_'+str(i)] = pagerank[i]
	sorted_x = sorted(ques_wt_dict.items(), key=operator.itemgetter(1))

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
	print args
	print ""
	main(args)
	
