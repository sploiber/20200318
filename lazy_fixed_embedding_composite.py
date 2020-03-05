from dwave.system.samplers import DWaveSampler
from dwave.system.composites import LazyFixedEmbeddingComposite

# Set up the QUBO. Start with the equations from the slides:
chainstrength = 4
numruns = 100
Q = {(0, 0): 2, (0, 1): -2, (0, 2): -2, (1, 2): 2}

sampler = LazyFixedEmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns)
print(sampler.properties['embedding'])

for smpl, energy in response.data(['sample', 'energy']):
    print(smpl, energy)
