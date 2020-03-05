import dimod
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite

Q = {(0, 0): 1, (3, 3): 1, (0, 1): -2, (1, 2): 2, (2, 3): -2}

chainstrength = 5
numruns = 1000

bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=-2)
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm, chain_strength=chainstrength, num_reads=numruns)
dwave.inspector.show(bqm, sampleset, sampler)
