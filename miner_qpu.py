from minorminer import find_embedding
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite

# Set up the QUBO. Start with the equations from the slides:
# x + y - 2xy -1
# 2yz - y - z
# -2zx + z + x - 1
# QUBO: 2x - 2xy + 2yz - 2zx - 2
Q = {(0, 0): 2, (0, 1): -2, (0, 2): -2, (1, 2): 2}

chainstrength = 2
numruns = 1000

dwave_sampler = DWaveSampler()
A = dwave_sampler.edgelist
embedding = find_embedding(Q, A)
print(embedding)

response = FixedEmbeddingComposite(DWaveSampler(), embedding).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)

for sample, energy, num, cbf in response.data(['sample', 'energy', 'num_occurrences', 'chain_break_fraction']):
    print(sample, energy, num, cbf)
