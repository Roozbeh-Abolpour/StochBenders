import time
import tracemalloc
import numpy as np

from stochbenders.milp.MILP import MILP
from stochbenders.solver.SBD_solver import sbd_solver
from stochbenders.master.master_separator import extract_masters
from stochbenders.subproblem.subproblem_separator import extract_subproblems


# ============================================================
# Deterministic test MILP
# Goal:
#   - exactly 2 masters
#   - exactly 2 subproblems per master
#   - optimal solution different from x=0, y=0
# ============================================================

nx = 4
ny = 4

# Objective:
# negative coefficients encourage positive x and y
c1 = np.array([-3.0, -2.0, -4.0, -1.0], dtype=float)
c2 = np.array([-2.0, -2.0, -2.0, -2.0], dtype=float)

# Inequalities A1 x + A2 y <= b
#
# Rows 0-1 connect x0,x1 with y0,y1   -> master group {0,1}, subproblem group {0,1}
# Rows 2-3 connect x2,x3 with y2,y3   -> master group {2,3}, subproblem group {2,3}
# Remaining rows are bounds on x only
A1 = np.array([
    [ 1.0,  1.0,  0.0,  0.0],   # x0 + x1 - 2y0 - 2y1 <= 0
    [ 1.0, -1.0,  0.0,  0.0],   # x0 - x1 <= 2

    [ 0.0,  0.0,  1.0,  1.0],   # x2 + x3 - 2y2 - 2y3 <= 0
    [ 0.0,  0.0,  1.0, -1.0],   # x2 - x3 <= 2

    [ 1.0,  0.0,  0.0,  0.0],   # x0 <= 3
    [ 0.0,  1.0,  0.0,  0.0],   # x1 <= 3
    [ 0.0,  0.0,  1.0,  0.0],   # x2 <= 3
    [ 0.0,  0.0,  0.0,  1.0],   # x3 <= 3

    [-1.0,  0.0,  0.0,  0.0],   # x0 >= 0
    [ 0.0, -1.0,  0.0,  0.0],   # x1 >= 0
    [ 0.0,  0.0, -1.0,  0.0],   # x2 >= 0
    [ 0.0,  0.0,  0.0, -1.0],   # x3 >= 0
], dtype=float)

A2 = np.array([
    [-2.0, -2.0,  0.0,  0.0],   # connects y0,y1
    [ 0.0,  0.0,  0.0,  0.0],

    [ 0.0,  0.0, -2.0, -2.0],   # connects y2,y3
    [ 0.0,  0.0,  0.0,  0.0],

    [ 0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0],

    [ 0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0],
], dtype=float)

b = np.array([
    0.0,
    2.0,
    0.0,
    2.0,
    3.0,
    3.0,
    3.0,
    3.0,
    0.0,
    0.0,
    0.0,
    0.0,
], dtype=float)

# No equalities
Ae1 = np.zeros((0, nx), dtype=float)
Ae2 = np.zeros((0, ny), dtype=float)
be = np.zeros(0, dtype=float)

milp = MILP(
    c1=c1,
    c2=c2,
    A1=A1,
    A2=A2,
    b=b,
    Ae1=Ae1,
    Ae2=Ae2,
    be=be,
)

# ============================================================
# Check decomposition structure
# ============================================================
master_milps, master_groups = extract_masters(milp)

print("=" * 70)
print("Decomposition structure")
print("=" * 70)
print("master groups:", master_groups)
print("number of masters:", len(master_milps))

for k, mm in enumerate(master_milps):
    sub_milps, sub_groups = extract_subproblems(mm)
    print(f"master {k} subproblem groups: {sub_groups}")
    print(f"master {k} number of subproblems: {len(sub_milps)}")

# Strong check: we want exactly 2 masters
if len(master_milps) != 2:
    raise AssertionError(f"Expected 2 masters, got {len(master_milps)}")

for k, mm in enumerate(master_milps):
    sub_milps, sub_groups = extract_subproblems(mm)
    if len(sub_milps) != 2:
        raise AssertionError(f"Expected 2 subproblems in master {k}, got {len(sub_milps)}")

# ============================================================
# Direct MILP solve
# ============================================================
print()
print("=" * 70)
print("Direct MILP solve")
print("=" * 70)

t0 = time.perf_counter()
x_ref, y_ref = milp.solve()
t_direct = time.perf_counter() - t0

obj_ref = c1 @ x_ref + c2 @ y_ref

print("x_ref =", x_ref)
print("y_ref =", y_ref)
print("obj_ref =", obj_ref)
print(f"direct solve time = {t_direct:.6f} seconds")

if np.linalg.norm(x_ref) <= 1e-10 and np.linalg.norm(y_ref) <= 1e-10:
    raise AssertionError("Direct MILP optimum is still x=0, y=0, but this test should be nonzero.")

# ============================================================
# Hierarchical SBD solve
# ============================================================
print()
print("=" * 70)
print("Hierarchical-layer SBD solve")
print("=" * 70)

tracemalloc.start()
t0 = time.perf_counter()

x_sbd, y_sbd = sbd_solver(milp)

t_sbd = time.perf_counter() - t0
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

obj_sbd = c1 @ x_sbd + c2 @ y_sbd

print("x_sbd =", x_sbd)
print("y_sbd =", y_sbd)
print("obj_sbd =", obj_sbd)
print(f"sbd solve time    = {t_sbd:.6f} seconds")
print(f"current memory    = {current_mem / 1024 / 1024:.6f} MB")
print(f"peak memory       = {peak_mem / 1024 / 1024:.6f} MB")

# ============================================================
# Comparison
# ============================================================
print()
print("=" * 70)
print("Comparison")
print("=" * 70)

x_match = np.linalg.norm(x_ref - x_sbd) <= 1e-6
y_match = np.linalg.norm(y_ref - y_sbd) <= 1e-6
obj_match = abs(obj_ref - obj_sbd) <= 1e-6

print("x match  :", x_match)
print("y match  :", y_match)
print("obj match:", obj_match)
print("||x_ref - x_sbd|| =", np.linalg.norm(x_ref - x_sbd))
print("||y_ref - y_sbd|| =", np.linalg.norm(y_ref - y_sbd))
print("|obj_ref - obj_sbd| =", abs(obj_ref - obj_sbd))

if x_match and y_match and obj_match:
    print()
    print("TEST PASSED: SBD solver matches direct MILP solve.")
else:
    raise AssertionError("TEST FAILED: SBD solver does not match direct MILP solve.")