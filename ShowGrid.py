# ShowGrid.py
import networkx as nx
import matplotlib.pyplot as plt
# 1-D
g1 = nx.grid_graph([10,])

# 2-D
g2 = nx.grid_graph([5, 5])

# 3-D
g3 = nx.grid_graph([3, 3, 3])

plt.figure()
nx.draw_kamada_kawai(g1)
plt.figure()
nx.draw_kamada_kawai(g2)
plt.figure()
nx.draw_kamada_kawai(g3)
plt.show()