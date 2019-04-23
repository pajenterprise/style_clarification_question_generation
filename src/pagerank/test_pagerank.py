import networkx as nx
import operator


if __name__ == "__main__":
    G = nx.DiGraph()
    N = 3 + 5
    for i in range(N):
        G.add_node(i)
    wt = 5

    G.add_edge(0, 3, weight=wt)
    #G.add_edge(3, 0, weight=wt)
    G.add_edge(0, 4, weight=wt)
    #G.add_edge(4, 0, weight=wt)
    G.add_edge(0, 5, weight=wt)
    #G.add_edge(5, 0, weight=wt)
    G.add_edge(1, 6, weight=wt)
    G#.add_edge(6, 1, weight=wt)
    G.add_edge(1, 4, weight=wt)
    #G.add_edge(4, 1, weight=wt)
    G.add_edge(2, 7, weight=wt)
    #G.add_edge(7, 2, weight=wt)

    G.add_edge(0, 1, weight=0.0085)
    G.add_edge(1, 0, weight=0.0085)

    G.add_edge(4, 6, weight=0.95)
    G.add_edge(6, 4, weight=0.95)
    G.add_edge(4, 7, weight=0.97)
    G.add_edge(7, 4, weight=0.97)

    pagerank = nx.pagerank(G)

    ques_wt_dict = {}
    for i in range(3, N):
        ques_wt_dict[i] = pagerank[i]

    sorted_x = sorted(ques_wt_dict.items(), key=operator.itemgetter(1))

    for curr_id, wt in sorted_x:
        print('%d: %.4f' % (curr_id, wt))