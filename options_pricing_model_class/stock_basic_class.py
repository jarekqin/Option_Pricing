from options_pricing_model_class.absstract_basic import BasicTreeOptionsClass
import numpy as np


class StockOPtion(object):
    def __init__(self,S0,K,r,T,N,pu,pd,div,sigma,is_put=False,is_am=False):
        """
        initialise the tree model of stock options
        :param S0: initial stock price
        :param K: strike price
        :param r: risk free rate
        :param T: maturity
        :param N: time steps
        :param pu: probability at up state
        :param pd: probability at down state
        :param div: dividend yield
        :param sigma: volatility on stock
        :param is_put: verify options is put or call
        :param is_am: verify options type is American options or European options
        """
        self.s0=S0
        self.k=K
        self.r=r
        self.t=T
        self.n=N
        self.sts=[]

        self.pu,self.pd=pu,pd
        self.div=div
        self.sigma=sigma
        self.is_call=not is_put
        self.is_european=not is_am

    @property
    def dt(self):
        return self.t/float(self.n)

    @property
    def df(self):
        return np.exp(-(self.r-self.div)*self.dt)


