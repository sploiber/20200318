import dimod
from minorminer import find_embedding
from dwave.embedding.chimera import find_clique_embedding
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from pyqubo import Array
import networkx as nx
import dwave_networkx as dnx
import matplotlib
from matplotlib import pyplot as plt


# Graph partitioning as full mesh
gamma = 3
N = 20

# Set up variables
x = Array.create('x', shape=N, vartype='BINARY')

H = gamma * (sum(x) - (N/2)) ** 2
for i in range(N):
    for j in range(i + 1, N):
        H += (x[i] - x[j]) ** 2

chainstrength = 5
numruns = 100

# Compile the model, and turn it into a QUBO object
model = H.compile()
Q, offset = model.to_qubo()
bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)

# Need to relabel variables for the first figure
bqm2 = bqm.relabel_variables({curr: v for v, curr in enumerate(bqm.variables)}, inplace=False)

# Do the embedding
dwave_sampler = DWaveSampler()
A = dwave_sampler.edgelist
embedding = find_embedding(Q, A)

# Draw the QUBO as a networkx graph
G = bqm2.to_networkx_graph()
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, font_size=10, node_size=150)
plt.show()

# Draw the embedded graph
G = dnx.chimera_graph(16, 16, 4)
dnx.draw_chimera_embedding(G, embedding, embedded_graph=bqm.to_networkx_graph(), unused_color=None)
plt.show()
