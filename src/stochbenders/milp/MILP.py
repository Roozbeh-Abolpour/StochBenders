import numpy as np
from pyomo.environ import *


class MILP:
    def __init__(self, A1,A2,b,Ae1,Ae2,be,c1,c2):
        self.A1=np.array(A1, dtype=float)
        self.A2=np.array(A2, dtype=float)
        self.b=np.array(b, dtype=float).reshape(-1)
        self.Ae1=np.array(Ae1, dtype=float)
        self.Ae2=np.array(Ae2, dtype=float)
        self.be=np.array(be, dtype=float).reshape(-1)
        self.c1=np.array(c1, dtype=float).reshape(-1)
        self.c2=np.array(c2, dtype=float).reshape(-1)
        self.check_dimensions()
        self.nx=self.c1.shape[0]
        self.ny=self.c2.shape[0]
        self.m=self.b.shape[0]
        self.me=self.be.shape[0]

    def check_dimensions(self):
        if self.A1.ndim != 2:
            raise ValueError("A1 must be 2D")
        if self.A2.ndim != 2:
            raise ValueError("A2 must be 2D")
        if self.Ae1.ndim != 2:
            raise ValueError("Ae1 must be 2D")
        if self.Ae2.ndim != 2:
            raise ValueError("Ae2 must be 2D")
        m1,n1 = self.A1.shape
        m2,n2 = self.A2.shape
        me1,ne1 = self.Ae1.shape
        me2,ne2 = self.Ae2.shape
        if m1 != m2 or m1 !=self.b.shape[0]:
            raise ValueError("A1, A2, and b must have the same number of rows")
        if me1 != me2 or me1 != self.be.shape[0]:
            raise ValueError("Ae1, Ae2, and be must have the same number of rows")
        if n1 != ne1 or n1 != self.c1.shape[0]:
            raise ValueError("A1, Ae1, and c1 must have the same number of columns") 
        if n2 != ne2 or n2 != self.c2.shape[0]:
            raise ValueError("A2, Ae2, and c2 must have the same number of columns")
    
    def solve(self):
        model=ConcreteModel()
        model.x=Var(range(self.nx), domain=Reals)
        model.y=Var(range(self.ny), domain=Binary)    
        model.obj=Objective(expr=sum(self.c1[i]*model.x[i] for i in range(self.nx))+sum(self.c2[j]*model.y[j] for j in range(self.ny)),sense=minimize)
        model.constraints=ConstraintList()  
        for i in range(self.m):
            model.constraints.add(sum(self.A1[i,j]*model.x[j] for j in range(self.nx))+sum(self.A2[i,j]*model.y[j] for j in range(self.ny))<=self.b[i]) 
        for i in range(self.me):
            model.constraints.add(sum(self.Ae1[i,j]*model.x[j] for j in range(self.nx))+sum(self.Ae2[i,j]*model.y[j] for j in range(self.ny))==self.be[i])
        solver=SolverFactory('highs')
        results=solver.solve(model, load_solutions=False)
        tc=results.solver.termination_condition
        if tc!=TerminationCondition.optimal:
            return None,None    
        model.solutions.load_from(results)
        x=np.array([value(model.x[i]) for i in range(self.nx)], dtype=float)
        y=np.array([value(model.y[j]) for j in range(self.ny)], dtype=float)
        return x,y
    
    def violation(self, x, y):
        v1=0.0
        v2=0.0
        if self.m>0:
            v1=np.max(self.A1 @ x+self.A2 @ y-self.b)
        if self.me>0:
            v2=np.max(np.abs(self.Ae1@x+self.Ae2 @ y-self.be))
        return max(0.0, v1, v2)
