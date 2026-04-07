
class DSU:
    def __init__(self,n):
        self.parent=[i for i in range(n)]
        self.rank=[0 for i in range(n)]
    
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        x=self.find(x)
        y=self.find(y)
        if x==y:
            return
        if self.rank[x]>self.rank[y]:
            self.parent[y]=x
        elif self.rank[y]>self.rank[x]:
            self.parent[x]=y
        else:
            self.parent[y]=x
            self.rank[x]+=1

    def groups(self):
        parent_map={}
        for i in range(len(self.parent)):
            p=self.find(i)
            if p not in parent_map:
                parent_map[p]=set()
            parent_map[p].add(i)
        return [sorted(g) for g in parent_map.values()]