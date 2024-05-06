"""
Examples with higher dimensions solved by GLIS

Note:
    - In this file, we solve the problem by
        - integrating/feeding the simulator/fun directly into the GLIS solver
        - with or without a user-defined nonlinear transformation of obj. fun.
    - Other formats are possible, check the file 'glis_1.py' for more details

Authors: A. Bemporad, M. Zhu
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from pyswarm import pso
from glis.solvers import GLIS

sys.path.append("C:/Users/s.music/Documents/CSI-algorithms")
from rastrigin import rastrigin

fixed_seed = 123
np.random.seed(fixed_seed)

nvars = 2
lb = -1.*np.ones(nvars)
ub = -lb
n_init = 10
n_max = 25

# compute optimum/optimizer by PSO
xopt0, fopt0 = pso(rastrigin, lb, ub, swarmsize=200, minfunc=1e-12, maxiter=50)

print("Solve the problem incrementally (i.e., provide the function evaluation at each iteration)")
prob = GLIS(bounds=(lb, ub), n_initial_random=10, alpha=1, delta=1)
x = prob.initialize()
for k in range(n_max):
    f = rastrigin(x)
    x = prob.update(f)
Xs = list(prob.X)
F = prob.F
xopt = prob.xbest
fopt = prob.fbest

# Plot
print("Optimization finished. Draw the plot")

X_r = np.arange(lb[0], ub[0]+0.1, 0.1)
Y_r = np.arange(lb[1], ub[1]+0.1, 0.1)

X, Y = np.meshgrid(X_r, Y_r)
Z = rastrigin([X, Y])

fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 600)
fig.colorbar(cf)
xs = np.squeeze(Xs)
plt.scatter(xs[:n_init,0], xs[:n_init,1], marker='o', c='orange', label='init samples')
plt.scatter(xs[n_init+1:,0], xs[n_init+1:,1], marker='o', label='active samples')
plt.scatter(xopt0[0], xopt0[1], c = 'r', marker='*', label='global optimum')
plt.scatter(xopt[0], xopt[1], marker='x', label='computed optimum')

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.legend()
plt.grid()

plt.savefig("glis_rastrigin.png", dpi=600)