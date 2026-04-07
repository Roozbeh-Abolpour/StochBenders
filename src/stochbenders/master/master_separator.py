import numpy as np
from stochbenders.milp.MILP import MILP
from stochbenders.dsu.dsu import DSU
from stochbenders.master.master import Master

def extract_masters(milp):       
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
    dsu=DSU(ny)
    for i in range(A1.shape[0]):        
        J={j for j in range(ny) if A2[i, j] != 0}
        if not J:
            continue
        first=next(iter(J))
        for j in J:
            dsu.union(first,j)
        
    for i in range(Ae1.shape[0]):
        J={j for j in range(ny) if Ae2[i, j] != 0}
        if not J:
            continue
        first=next(iter(J))
        for j in J:
            dsu.union(first,j)

    groups=dsu.groups()
    milps=[]    
    for g in groups:       
        not_g=[i for i in range(ny) if i not in g]
        L=[i for i in range(m) if  all(A2[i,j]==0 for j in not_g)] 
        M=[j for j in g]
        A1_e=A1[L,:]
        A2_e=A2[L,:][:,M]

        Le=[i for i in range(me) if  all(Ae2[i,j]==0 for j in not_g)] 
        Ae1_e=Ae1[Le,:]
        Ae2_e=Ae2[Le,:][:,M]
        c1_e=milp.c1[:]
        c2_e=milp.c2[M] 
        b_e=b[L]
        be_e=be[Le]    
        milp_e=MILP(A1_e,A2_e,b_e,Ae1_e,Ae2_e,be_e,c1_e,c2_e)    
        milps.append(milp_e)

    return milps,groups

def aggregate_masters_inf_cuts(masters,groups):
    ny=max(max(g) for g in groups)+1
    inf_cuts=list()
    for k,master in enumerate(masters):
        for cut in master.inf_cuts:
            temp_cut=np.zeros(ny+1)        
            temp_cut[groups[k]]=cut[:-1]
            temp_cut[-1]=cut[-1]
            inf_cuts.append(temp_cut)
    return inf_cuts