import numpy as np
from stochbenders.master.master import Master
from stochbenders.subproblem.subproblem import Subproblem
from stochbenders.master.master_separator import extract_masters,aggregate_masters_inf_cuts
from stochbenders.subproblem.subproblem_separator import extract_subproblems,pool_subproblems,aggregate_subproblems_solutions

def sbd_solver(milp):        
    master_milps,master_groups=extract_masters(milp)        
    masters=[Master(m) for m in master_milps]    
    local_subproblem_milps={}
    for master, master_milp in zip(masters,master_milps):        
        subproblems,groups=extract_subproblems(master_milp)
        local_subproblem_milps[master]=subproblems
    
    local_subproblems={mas:[Subproblem(milpe) for milpe in local_subproblem_milps[mas]] for mas in masters}
    flags={master: True for master in masters}
    while any(flags.values()):
        for master in masters:
            if not flags[master]:
                continue            
            y=master.solve()    
            for subproblem in local_subproblems[master]:
                x,ef=subproblem.solve(y)            
                if ef==1:
                    subproblem.inf_cut(y)                                    
                elif ef==0:
                    subproblem.opt_cut(y)
                
            cuts,flag=pool_subproblems(local_subproblems[master])
            if not flag:
                for cut in cuts:
                    master.add_inf_cut(cut)                
            else:
                flags[master]=False
    
    milp_master=Master(milp)
    inf_cuts=aggregate_masters_inf_cuts(masters,master_groups)
    for cut in inf_cuts:        
        milp_master.add_inf_cut(cut)
        
    global_subproblems_milps,groups=extract_subproblems(milp)
    subproblems=[Subproblem(m) for m in global_subproblems_milps]
    yb=None
    xb=None
    while True:
        xs=[]
        y=milp_master.solve()        
        for subproblem in subproblems:
            x,ef=subproblem.solve(y)            
            if ef==0:
                subproblem.opt_cut(y)
                xs.append(x)                               
            elif ef==1:
                subproblem.inf_cut(y)                
            else:
                raise Exception('Subproblem is unbounded or infeasible or both. Check the model and the implementation of the subproblem.')
        cuts,flag=pool_subproblems(subproblems)
        if not flag:
            for cut in cuts:
                milp_master.add_inf_cut(cut)                                         
            x=None    
        else:
            milp_master.add_opt_cut(cuts[0])
            x=aggregate_subproblems_solutions(xs,groups)
        if len(xs)==len(subproblems) and xb is not None and x is not None and yb is not None and y is not None  and np.linalg.norm(y-yb)+np.linalg.norm(x-xb)<=1e-8:
            break
        yb=y    
        xb=x
    x=np.array(x, dtype=float)
    y=np.array(y, dtype=float)
    return x,y


