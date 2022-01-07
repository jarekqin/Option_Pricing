import numpy as np

from options_pricing_model_class.stock_basic_class import StockOPtion
from options_pricing_model_class.absstract_basic import BasicTreeOptionsClass


class BinomialEuropeanOption(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        self.m = self.n + 1
        self.u = 1 + self.pu
        self.d = 1 - self.pd
        self.qu = (np.exp(
            (self.r - self.div) * self.dt) - self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def init_stock_price_tree(self):
        self.sts = np.zeros(self.m)
        for i in range(self.m):
            self.sts[i] = self.s0 * (self.u ** (self.n - i)) * (self.d ** i)

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.sts - self.k)
        else:
            return np.maximum(0, self.k - self.sts)

    def traverse_tree(self, payoffs):
        for i in range(self.n):
            payoffs = (payoffs[:-1] * self.qu + payoffs[1:] * self.qd) * self.df
        return payoffs

    def begine_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begine_tree_traversal()
        return payoffs[0]


class BinomialAmericanOption(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        self.m = self.n + 1
        self.u = 1 + self.pu
        self.d = 1 - self.pd
        self.qu = (np.exp(
            (self.r - self.div) * self.dt) - self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def init_stock_price_tree(self):
        self.sts = [np.array([self.s0])]
        for i in range(self.n):
            prev_branches = self.sts[-1]
            st = np.concatenate(
                (prev_branches * self.u, [prev_branches[-1] * self.d])
            )
            self.sts.append(st)

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.sts[self.n] - self.k)
        else:
            return np.maximum(0, self.k - self.sts[self.n])

    def check_early_exercise(self, payoffs, node):
        if self.is_call:
            return np.maximum(payoffs, self.sts[node] - self.k)
        else:
            return np.maximum(payoffs, self.k - self.sts[node])

    def traverse_tree(self, payoffs):
        for i in reversed(range(self.n)):
            payoffs = (payoffs[:-1] * self.qu + payoffs[1:] * self.qd) * self.df
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs, i)
        return payoffs

    def begine_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begine_tree_traversal()
        return payoffs[0]


class BinomialCRROption(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        self.u = np.exp(self.sigma*np.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (np.exp(
            (self.r - self.div) * self.dt) - self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def init_stock_price_tree(self):
        self.sts = [np.array([self.s0])]
        for i in range(self.n):
            prev_branches = self.sts[-1]
            st = np.concatenate(
                (prev_branches * self.u, [prev_branches[-1] * self.d])
            )
            self.sts.append(st)

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.sts[self.n] - self.k)
        else:
            return np.maximum(0, self.k - self.sts[self.n])

    def check_early_exercise(self, payoffs, node):
        if self.is_call:
            return np.maximum(payoffs, self.sts[node] - self.k)
        else:
            return np.maximum(payoffs, self.k - self.sts[node])

    def traverse_tree(self, payoffs):
        for i in reversed(range(self.n)):
            payoffs = (payoffs[:-1] * self.qu + payoffs[1:] * self.qd) * self.df
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs, i)
        return payoffs

    def begine_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begine_tree_traversal()
        return payoffs[0]

if __name__ == '__main__':
    model = BinomialEuropeanOption(50, 52, r=0.05, t=2, n=2, pu=0.2, pd=0.2, is_put=True)
    print(model.price())
    model = BinomialAmericanOption(50, 52, r=0.05, t=2, n=2, pu=0.2, pd=0.2, is_put=True, is_am=True)
    print(model.price())
    model=BinomialCRROption(50,52,r=0.05,t=2,n=2,sigma=0.3,is_put=True)
    print(model.price())