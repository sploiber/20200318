# Copyright 2020 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dimod
from minorminer import find_embedding
from dwave.embedding.chimera import find_clique_embedding
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from pyqubo import Array
import networkx as nx
import dwave_networkx as dnx
import matplotlib
#matplotlib.use('Qt4Agg', warn=True)
from matplotlib import pyplot as plt


# Graph partitioning as full mesh
gamma = 3
N = 8

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
