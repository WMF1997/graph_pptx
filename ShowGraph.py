# ShowGraph.py
# 2019.08.07

# 展示的图是: Karate俱乐部的图. 

import networkx as nx
import matplotlib.pyplot as plt

karate = nx.karate_club_graph()


fig1 = plt.figure()
nx.draw_kamada_kawai(karate, with_labels=True, node_color='#66CCFF', edge_color='#EE0000', font_color='#006666')

fig2 = plt.figure()
karate2 = karate.copy().to_directed()
nx.draw_kamada_kawai(karate2, with_labels=True, node_color='#66CCFF', edge_color='#EE0000', font_color='#006666')

plt.show()
