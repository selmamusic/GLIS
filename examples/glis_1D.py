"""
1D example solved using GLIS (for visualization)
    - with box constraints only

Note:
    - Problems can be solved by
        - integrating/feeding the simulator/fun directly into the GLIS solver
            - simulator/fun:
                - input: a sample to test (provided by GLIS)
                - output: the evaluation
            - the intermediate steps within the simulator/fun are unknown to the GLIS (black-box)
        - incrementally (i.e., provide the function evaluation at each iteration)

Authors: A. Bemporad, M. Zhu, S. Music
"""

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from glis.solvers import GLIS
from math import sin, cos
from itertools import chain

benchmark = "scalar_bemporad20"

savefigs = True

if benchmark == "scalar_bemporad20":
    # 1D function from bemporad20 paper
    lb = -3.
    ub = 3.
    fun = lambda x: ((1 + (x[0] * sin(2*x[0]) * cos(3*x[0])) / (1 + x[0]**2))**2 + x[0]**2 / 12 + x[0] / 10)
    funs = lambda x: ((1 + (x * sin(2*x) * cos(3*x)) / (1 + x**2))**2 + x**2 / 12 + x / 10)
    xopt0 = -0.9599  # unconstrained optimizer
    fopt0 = 0.2795  # unconstrained optimum
    max_evals = 20

key = 1
np.random.seed(key)  # rng default for reproducibility
####################################################################################
print("Solve the problem incrementally (i.e., provide the function evaluation at each iteration)")
# solve same problem, but incrementally
prob = GLIS(bounds=(lb, ub), n_initial_random=10)
x = prob.initialize()
f_surr = np.array([])
for k in range(max_evals):
    f = fun(x)
    x = prob.update(f)
    f_surr = np.append(f_surr, prob.F[k])
X = list(chain.from_iterable(prob.X[:-1]))  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated at max_evals +1)
F = prob.F
xopt = prob.xbest
fopt = prob.fbest
##########################################

# Plot
print("Optimization finished. Draw the plot")

plt.rcParams['text.usetex'] = True
X_r = np.arange(lb, ub, 0.1)
F_r = [funs(i) for i in X_r]

plt.plot(X_r, F_r, label=r'$f(x)$')
plt.plot(X[:9], F[:9], "o", label=r'init samples')
plt.plot(X[10:], F[10:], "o", color = (0.5, 0.8, 0.9), label=r'glis samples')
plt.plot(xopt0, fopt0, "*", label=r'real optimum')
plt.plot(xopt, fopt, "x", label=r'computed optimum')

plt.xlim(lb, ub)
plt.xlabel(r'$x$')
plt.legend(fontsize=12)
plt.grid()

if savefigs:
    plt.savefig("glis-1D.png", dpi=300)
plt.show()