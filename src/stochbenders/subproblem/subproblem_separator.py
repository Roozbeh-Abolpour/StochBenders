import numpy as np
from stochbenders.milp.MILP import MILP
from stochbenders.dsu.dsu import DSU
from stochbenders.subproblem.subproblem import Subproblem

def extract_subproblems(milp):       
    A1=milp.A1
    A2=milp.A2
    Ae1=milp.Ae1
    Ae2=milp.Ae2
    b=milp.b
    be=milp.be
    nx=milp.nx
    ny=milp.ny
    m=milp.m
    me=milp.me    
    dsu=DSU(nx)
    for i in range(A1.shape[0]):        
        J={j for j in range(nx) if A1[i, j] != 0}
        if not J:
            continue
        first=next(iter(J))
        for j in J:
            dsu.union(first,j)
        
    for i in range(Ae1.shape[0]):
        J={j for j in range(nx) if Ae1[i, j] != 0}
        if not J:
            continue
        first=next(iter(J))
        for j in J:
            dsu.union(first,j)

    groups=dsu.groups()
    milps=[]    
    for g in groups:        
        M=[j for j in g]
        L=[j for j in range(m) if np.any(A1[j,M]!=0)];        
        A1_e=A1[L,:][:,M]        
        A2_e=A2[L,:]

        Le=[j for j in range(me) if np.any(Ae1[j,M]!=0)];                
        Ae1_e=Ae1[Le,:][:,M]
        Ae2_e=Ae2[Le,:]
        c1_e=milp.c1[M]
        c2_e=milp.c2[:] 
        b_e=b[L]
        be_e=be[Le]    
        milp_e=MILP(A1_e,A2_e,b_e,Ae1_e,Ae2_e,be_e,c1_e,c2_e)    
        milps.append(milp_e)
    return milps,groups

def pool_subproblems(subproblems):   
    inf_cuts=[subproblem.current_inf_cut for subproblem in subproblems if subproblem.current_inf_cut is not None]        
    if len(inf_cuts)>0:        
        return inf_cuts,False
    else:        
        opt_cut=np.zeros_like(subproblems[0].current_opt_cut)
        for subproblem in subproblems:
            opt_cut+=subproblem.current_opt_cut
        return [opt_cut],True

def aggregate_subproblems_solutions(xs, groups):
    nx=max(max(g) for g in groups)+1
    x=np.zeros(nx)
    for i, g in enumerate(groups):
        G = sorted(list(g))
        for k, j in enumerate(G):
            x[j] = xs[i][k]
    return x