from pyomo.environ import *
import numpy as np

class Master:
    def __init__(self, milp):
        self.opt_cuts=np.array([]).reshape(0,milp.ny+1)
        self.inf_cuts=np.array([]).reshape(0,milp.ny+1)
        A1=milp.A1
        A2=milp.A2
        Ae1=milp.Ae1    
        Ae2=milp.Ae2        
        b=milp.b
        be=milp.be
        m=milp.m
        me=milp.me
        J=[j for j in range(m) if np.all(A1[j,:]==0)]
        if not J:
            self.base_A=np.zeros((0,milp.ny))
            self.base_b=np.zeros(0)
        else:
            self.base_A=A2[J,:]
            self.base_b=b[J]
        Je=[j for j in range(me) if np.all(Ae1[j,:]==0)]
        if not Je:
            self.base_Aeq=np.zeros((0,milp.ny))
            self.base_beq=np.zeros(0)
        else:
            self.base_Aeq=Ae2[Je,:]
            self.base_beq=be[Je]
        self.c=milp.c2

    def add_opt_cut(self,cut):
        cut=np.array(cut).reshape(1,-1)
        self.opt_cuts=np.vstack([self.opt_cuts,cut])

    def add_inf_cut(self,cut):
        cut=np.array(cut).reshape(1,-1)
        self.inf_cuts=np.vstack([self.inf_cuts,cut])

    def solve(self):
        c=self.c
        n=len(c)
        model=ConcreteModel()
        model.x=Var(range(n), domain=Binary)
        model.theta=Var(domain=Reals,bounds=(-1e6,None))
        model.obj=Objective(expr=model.theta+sum(c[i]*model.x[i] for i in range(n)),sense=minimize)
        model.constraints=ConstraintList()
        for i in range(len(self.base_b)):
            model.constraints.add(sum(self.base_A[i, j]*model.x[j] for j in range(n))<=self.base_b[i])
        for i in range(len(self.base_beq)):
            model.constraints.add(sum(self.base_Aeq[i, j]*model.x[j] for j in range(n))==self.base_beq[i])
        for cut in self.opt_cuts:
            model.constraints.add(sum(cut[j]*model.x[j] for j in range(n))+cut[n]<=model.theta)                
        for cut in self.inf_cuts:            
            model.constraints.add(sum(cut[j]*model.x[j] for j in range(n))+cut[n]<=0)
            
        solver=SolverFactory('highs')                
        results=solver.solve(model, load_solutions=False)
        tc=results.solver.termination_condition
        if tc!=TerminationCondition.optimal:
            return None
        model.solutions.load_from(results)
        out=np.array([value(model.x[i]) for i in range(n)])
        return out
    
    def __str__(self):
        return f"Master with base A:\n{self.base_A}\nbase b:\n{self.base_b}\nbase Aeq:\n{self.base_Aeq}\nbase beq:\n{self.base_beq}\nc:\n{self.c}\nopt cuts:\n{self.opt_cuts}\ninf cuts:\n{self.inf_cuts}"