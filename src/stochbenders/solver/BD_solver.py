import numpy as np
from stochbenders.master.master import Master
from stochbenders.subproblem.subproblem import Subproblem

def bd_solver(milp):        
    milp_master=Master(milp)
    subproblem=Subproblem(milp)
    xb=None
    yb=None

    while True:
        y=milp_master.solve()        
        x,ef=subproblem.solve(y)            
        if ef==0:
            cut=subproblem.opt_cut(y)            
            milp_master.add_opt_cut(cut)
        elif ef==1:
            cut=subproblem.inf_cut(y)                
            milp_master.add_inf_cut(cut)
        else:
            raise Exception('Subproblem is unbounded or infeasible or both. Check the model and the implementation of the subproblem.')
        
        if xb is not None and x is not None and yb is not None and y is not None  and np.linalg.norm(y-yb)+np.linalg.norm(x-xb)<=1e-8:
            break
        yb=y    
        xb=x
    x=np.array(x, dtype=float)
    y=np.array(y, dtype=float)
    return x,y


