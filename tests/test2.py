import time
import numpy as np
from stochbenders.milp.MILP import MILP
from stochbenders.solver.SBD_solver import sbd_solver
from stochbenders.solver.BD_solver import bd_solver

def violation(milp, x, y):
    v = 0.0
    if milp.m > 0:
        v = max(v, float(np.max(milp.A1 @ x + milp.A2 @ y - milp.b)))
    if milp.me > 0:
        v = max(v, float(np.max(np.abs(milp.Ae1 @ x + milp.Ae2 @ y - milp.be))))
    return max(0.0, v)

# -------------------------------
# simple structured MILP
# -------------------------------

# variables
# x = (x1, x2, x3)
# y = (y1, y2)

c1 = np.array([-3.0, -2.0, -1.0])
c2 = np.array([2.0, 1.0])

# inequalities
A1 = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
], dtype=float)

A2 = np.array([
    [-2, 0],
    [0, -2],
    [-1, -1]
], dtype=float)

b = np.array([2.0, 2.0, 2.0])

# equalities (simple coupling)
Ae1 = np.array([
    [1, -1, 0]
], dtype=float)

Ae2 = np.array([
    [1, -1]
], dtype=float)

be = np.array([0.0])

for i in range(3):
    row = np.zeros(3)
    row[i] = -1.0
    A1 = np.vstack((A1, row))
    A2 = np.vstack((A2, np.zeros(2)))
    b = np.append(b, 0.0)

    row = np.zeros(3)
    row[i] = 1.0
    A1 = np.vstack((A1, row))
    A2 = np.vstack((A2, np.zeros(2)))
    b = np.append(b, 3.0)

milp = MILP(c1=c1, c2=c2, A1=A1, A2=A2, b=b, Ae1=Ae1, Ae2=Ae2, be=be)

# -------------------------------
# solve
# -------------------------------

t0 = time.perf_counter()
x0, y0 = milp.solve()
t_highs = time.perf_counter() - t0

t0 = time.perf_counter()
x1, y1 = sbd_solver(milp)
t_sbd = time.perf_counter() - t0

t0 = time.perf_counter()
x2, y2 = bd_solver(milp)
t_bd = time.perf_counter() - t0

x0 = np.array(x0, dtype=float)
y0 = np.array(y0, dtype=float)
x1 = np.array(x1, dtype=float)
y1 = np.array(y1, dtype=float)
x2 = np.array(x2, dtype=float)
y2 = np.array(y2, dtype=float)

obj0 = float(c1 @ x0 + c2 @ y0)
obj1 = float(c1 @ x1 + c2 @ y1)
obj2 = float(c1 @ x2 + c2 @ y2)

print("obj_highs =", obj0)
print("obj_sbd   =", obj1)
print("obj_bd    =", obj2)
print("gap_sbd   =", abs(obj0 - obj1))
print("gap_bd    =", abs(obj0 - obj2))
print("")
print("vio_highs =", violation(milp, x0, y0))
print("vio_sbd   =", violation(milp, x1, y1))
print("vio_bd    =", violation(milp, x2, y2))
print("")
print("time_highs =", t_highs)
print("time_sbd   =", t_sbd)
print("time_bd    =", t_bd)