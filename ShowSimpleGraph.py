# ShowSimpleGraph.py
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
np.random.seed(19970121)
random.seed(19970121)
G = nx.erdos_renyi_graph(10, 0.7).to_directed()
edges_to_remove = random.sample(G.edges(), int(G.number_of_edges()*0.5))
G.remove_edges_from(edges_to_remove)

# nx.draw_kamada_kawai(G, with_labels=True)
# plt.show()
