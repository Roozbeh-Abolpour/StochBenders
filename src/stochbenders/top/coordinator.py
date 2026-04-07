import numpy as np
from stochbenders.milp.MILP import MILP
from stochbenders.dsu.dsu import DSU

def decompose_milp(milp):
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
    groups=[]
    dsu=DSU(nx+ny)
    for i in range(A1.shape[0]):        
        J = {j for j in range(nx) if A1[i, j] != 0} | {nx + j for j in range(ny) if A2[i, j] != 0}
        if not J:
            continue
        first=next(iter(J))
        for j in J:
            dsu.union(first,j)
        
    for i in range(Ae1.shape[0]):
        J = {j for j in range(nx) if Ae1[i, j] != 0} | {nx + j for j in range(ny) if Ae2[i, j] != 0}
        if not J:
            continue
        first=next(iter(J))
        for j in J:
            dsu.union(first,j)

    groups=dsu.groups()
    milps=[]    
    for g in groups:
        M1=[j for j in g if j<nx]
        M2=[j-nx for j in g if j>=nx]
        L=[j for j in range(m) if np.any(A1[j,M1]!=0) or np.any(A2[j,M2]!=0)];        
        A1_e=A1[L,:][:,M1]        
        A2_e=A2[L,:][:,M2]

        Le=[j for j in range(me) if np.any(Ae1[j,M1]!=0) or np.any(Ae2[j,M2]!=0)];                
        Ae1_e=Ae1[Le,:][:,M1]
        Ae2_e=Ae2[Le,:][:,M2]
        c1_e=milp.c1[M1]
        c2_e=milp.c2[M2]        
        b_e=b[L]
        be_e=be[Le]        
        milps.append(MILP(A1_e,A2_e,b_e,Ae1_e,Ae2_e,be_e,c1_e,c2_e))

    return milps