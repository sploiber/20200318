import dimod

# use the exact solver to find energies for all states. This is only
# realistic for very small problems.
exactsolver = dimod.ExactSolver()

Q = {(0, 0): 1, (0, 1): -1}

results = exactsolver.sample_qubo(Q)

# print the results
for smpl, energy in results.data(['sample', 'energy']):
    print(smpl, energy)
