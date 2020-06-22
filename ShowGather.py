# ShowGather.py
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

h = 3
w = 3
nodes = np.arange(h*w)
center_node = int(np.median(nodes))

gather_edges = np.array([[i, center_node] for i in nodes])
positions = np.array([[i, j] for j in range(w) for i in range(h)])
g33 = nx.DiGraph()
g33.add_nodes_from(nodes)
g33.add_edges_from(gather_edges)

node_color = np.array(['#66CCFF']*(h*w))
node_color[center_node] = '#AA667F'
edge_color = '#EE0000'
font_color = '#006666'

plt.figure()
nx.draw(g33, pos=positions, with_labels=True, node_color=node_color, edge_color=edge_color, font_color=font_color)
plt.show()