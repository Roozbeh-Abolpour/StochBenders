from pyomo.environ import *
import numpy as np

class Subproblem:
    def __init__(self, milp):
        self.A1=milp.A1
        self.A2=milp.A2
        self.Ae1=milp.Ae1
        self.Ae2=milp.Ae2
        self.b=milp.b
        self.be=milp.be
        self.c1=milp.c1
        self.c2=milp.c2
        self.current_inf_cut=None
        self.current_opt_cut=None



    def solve(self,y):
        A=self.A1
        b=self.b-self.A2@y
        Ae=self.Ae1
        be=self.be-self.Ae2@y
        c=self.c1
        n=len(c)
        model=ConcreteModel()
        model.x=Var(range(n), domain=Reals)        
        model.obj=Objective(expr=sum(c[i]*model.x[i] for i in range(n)),sense=minimize)
        model.constraints=ConstraintList()
        for i in range(len(b)):
            model.constraints.add(sum(A[i,j]*model.x[j] for j in range(n))<=b[i])
        for i in range(len(be)):
            model.constraints.add(sum(Ae[i,j]*model.x[j] for j in range(n))==be[i])        
        solver=SolverFactory('highs')        
        results=solver.solve(model, load_solutions=False)
        tc=results.solver.termination_condition                
        if tc==TerminationCondition.optimal:
            exit_flag=0
            model.solutions.load_from(results)
            out=np.array([value(model.x[i]) for i in range(n)], dtype=float)
            return out, exit_flag        
        elif tc==TerminationCondition.infeasible:
            exit_flag=1
            return None, exit_flag
        elif tc==TerminationCondition.unbounded:
            exit_flag=2 
            return None, exit_flag
        elif tc==TerminationCondition.infeasibleOrUnbounded:
            exit_flag=3 
            return None, exit_flag
        else:
            exit_flag=4
            return None, exit_flag
    
    
    def opt_cut(self,y):
        A1=self.A1
        A2=self.A2
        Ae1=self.Ae1
        Ae2=self.Ae2
        b=self.b
        be=self.be
        c1=self.c1
        c2=self.c2
        m=len(b)
        me=len(be)
        model=ConcreteModel()
        model.lamda=Var(range(m), domain=Reals, bounds=(0,None))
        model.mu=Var(range(me), domain=Reals)        
        bb=b-A2@y
        bbe=be-Ae2@y
        model.obj=Objective(expr=-sum(model.lamda[i]*bb[i] for i in range(m))-sum(model.mu[i]*bbe[i] for i in range(me)),sense=maximize)
        model.constraints=ConstraintList()
        for i in range(len(c1)):
            model.constraints.add(sum(A1[j,i]*model.lamda[j] for j in range(m))+sum(Ae1[j,i]*model.mu[j] for j in range(me))==-c1[i])
        solver=SolverFactory('highs')        
        results=solver.solve(model, load_solutions=False)
        tc=results.solver.termination_condition
        if tc!=TerminationCondition.optimal:
            return None
        model.solutions.load_from(results)
        lamda=np.array([value(model.lamda[i]) for i in range(m)], dtype=float)
        mu=np.array([value(model.mu[i]) for i in range(me)], dtype=float)
        g=A2.T@lamda+Ae2.T@mu
        h=sum(lamda[i]*b[i] for i in range(m))+sum(mu[i]*be[i] for i in range(me))
        cut=np.transpose(g)
        cut=np.hstack([cut,-h])
        self.current_opt_cut=cut
        self.current_inf_cut=None
        return cut
    

    def inf_cut(self,y):
        A1=self.A1
        A2=self.A2
        Ae1=self.Ae1
        Ae2=self.Ae2
        b=self.b
        be=self.be
        c1=self.c1
        m=len(b)
        me=len(be)
        model=ConcreteModel()
        model.lamda=Var(range(m), domain=Reals, bounds=(0,None))
        model.mu=Var(range(me), domain=Reals)
        bb=b-A2@y
        bbe=be-Ae2@y
        model.obj=Objective(expr=0,sense=minimize)
        model.constraints=ConstraintList()
        for i in range(len(c1)):
            model.constraints.add(sum(A1[j,i]*model.lamda[j] for j in range(m))+sum(Ae1[j,i]*model.mu[j] for j in range(me))==0)        
        model.constraints.add(sum(model.lamda[i]*bb[i] for i in range(m))+sum(model.mu[i]*bbe[i] for i in range(me))==-1)        
        solver=SolverFactory('highs')                
        results=solver.solve(model, load_solutions=False)
        tc=results.solver.termination_condition
        if tc!=TerminationCondition.optimal:
            return None
        model.solutions.load_from(results)
        lamda=np.array([value(model.lamda[i]) for i in range(m)], dtype=float)
        mu=np.array([value(model.mu[i]) for i in range(me)], dtype=float)
        g=A2.T@lamda+Ae2.T@mu
        h=sum(lamda[i]*b[i] for i in range(m))+sum(mu[i]*be[i] for i in range(me))
        cut=np.transpose(g)
        cut=np.hstack([cut,-h])
        self.current_inf_cut=cut
        self.current_opt_cut=None
        return cut
    
    def __str__(self):
        return f"Subproblem with A1:\n{self.A1}\nA2:\n{self.A2}\nAe1:\n{self.Ae1}\nAe2:\n{self.Ae2}\nb:\n{self.b}\nbe:\n{self.be}\nc1:\n{self.c1}\nc2:\n{self.c2}"