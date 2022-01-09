import numpy as np


class FiniteDifferenceBasicClass(object):
    def __init__(self,s0,k,r=0.05,t=1,sigma=0,smax=1,m=1,n=1,is_put=False):
        self.s0=s0
        self.k=k
        self.r=r
        self.sigma=sigma
        self.smax=smax
        self.t=t
        self.m=m
        self.n=n
        self.is_call=not is_put
        self.i_values=np.arange(self.m)
        self.j_values=np.arange(self.n)
        self.grid=np.zeros((self.m+1,self.n+1))
        self.boundary_conds=np.linspace(0,smax,self.m+1)


    @property
    def ds(self):
        return self.smax/float(self.m)

    @property
    def dt(self):
        return self.t/float(self.m)
