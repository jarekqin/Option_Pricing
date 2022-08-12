import numpy as np

from vanilla_options_model.basic_class.stock_basic_class import StockOPtion
from vanilla_options_model.basic_class.abstract_basic import BasicTreeOptionsClass


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
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1. / self.u
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


class BinomialLROption(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        odd_n = self.n if self.n % 2 == 0 else self.n + 1
        d1 = (np.log(self.s0 / self.k) + ((self.r - self.div) + (self.sigma ** 2) / 2.) * self.t) / \
             (self.sigma * np.sqrt(self.t))
        d2 = (np.log(self.s0 / self.k) + ((self.r - self.div) - (self.sigma ** 2) / 2.) * self.t) / \
             (self.sigma * np.sqrt(self.t))
        pbar = self.pp_2_inversion(d1, odd_n)
        self.p = self.pp_2_inversion(d2, odd_n)
        self.u = 1 / self.df * pbar / self.p
        self.d = (1 / self.df - self.p * self.u) / (1 - self.p)
        self.qu = self.p
        self.qd = 1 - self.p

    def pp_2_inversion(self, z, n):
        return .5 + np.copysign(1, z) * \
               np.sqrt(.25 - .25 * np.exp(
                   -((z / (n + 1. / 3. + .1 / (n + 1))) ** 2.) * (n + 1 / 6.)
               )
                       )

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


class BinomialLROptionGreeks(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        odd_n = self.n if self.n % 2 == 0 else self.n + 1
        d1 = (np.log(self.s0 / self.k) + ((self.r - self.div) + (self.sigma ** 2) / 2.) * self.t) / \
             (self.sigma * np.sqrt(self.t))
        d2 = (np.log(self.s0 / self.k) + ((self.r - self.div) - (self.sigma ** 2) / 2.) * self.t) / \
             (self.sigma * np.sqrt(self.t))
        pbar = self.pp_2_inversion(d1, odd_n)
        self.p = self.pp_2_inversion(d2, odd_n)
        self.u = 1 / self.df * pbar / self.p
        self.d = (1 / self.df - self.p * self.u) / (1 - self.p)
        self.qu = self.p
        self.qd = 1 - self.p

    def pp_2_inversion(self, z, n):
        return .5 + np.copysign(1, z) * \
               np.sqrt(.25 - .25 * np.exp(
                   -((z / (n + 1. / 3. + .1 / (n + 1))) ** 2.) * (n + 1 / 6.)
               )
                       )

    def new_stock_price_tree(self):
        self.sts = [np.array([self.s0 * self.u / self.d, self.s0, self.s0 * self.d / self.u])]
        for i in range(self.n):
            prev_branches = self.sts[-1]
            st = np.concatenate((prev_branches * self.u, [prev_branches[-1] * self.d]))
            self.sts.append(st)

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
        self.new_stock_price_tree()
        payoffs = self.begine_tree_traversal()
        option_value = payoffs[len(payoffs) // 2]
        payoff_up = payoffs[0]
        payoff_down = payoffs[-1]
        s_up = self.sts[0][0]
        s_down = self.sts[0][-1]
        ds_up = s_up - self.s0
        ds_down = self.s0 - s_down
        ds = s_up - s_down
        dv = payoff_up - payoff_down
        delta = dv / ds
        gamma = ((payoff_up - option_value) / ds_up - (option_value - payoff_down) / ds_down) / \
                ((self.s0 + s_up) / 2. - (self.s0 + s_down) / 2.)

        return option_value, delta, gamma


class TrinomialLROption(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        self.u = np.exp(self.sigma * np.sqrt(2. * self.dt))
        self.d = 1 / self.u
        self.m = 1
        self.qu = ((np.exp((self.r - self.div) * self.dt / 2.) -
                    np.exp(-self.sigma * np.sqrt(self.dt / 2.))) /
                   (np.exp(self.sigma * np.sqrt(self.dt / 2.)) - np.exp(
                       -self.sigma * np.sqrt(self.dt / 2.)))) ** 2
        self.qd = ((np.exp(self.sigma * np.sqrt(self.dt / 2.)) -
                    np.exp((self.r - self.div) * self.dt / 2.)) /
                   (np.exp(self.sigma * np.sqrt(self.dt / 2.)) -
                    np.exp(-self.sigma * np.sqrt(self.dt / 2.))
                    )) ** 2.

        self.qm = 1 - self.qu - self.qd

    def init_stock_price_tree(self):
        self.sts = [np.array([self.s0])]
        for i in range(self.n):
            prev_nodes = self.sts[-1]
            self.st = np.concatenate(
                (prev_nodes * self.u, [prev_nodes[-1] * self.m, prev_nodes[-1] * self.d])
            )
            self.sts.append(self.st)

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
            payoffs = (payoffs[:-2] * self.qu + payoffs[1:-1] * self.qm + payoffs[2:] * self.qd) * self.df
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


class BinomialCRROptionLattice(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1. / self.u
        self.qu = (np.exp(
            (self.r - self.div) * self.dt) - self.d) / (self.u - self.d)
        self.qd = 1 - self.qu
        self.m = 2 * self.n + 1

    def init_stock_price_tree(self):
        self.sts = np.zeros(self.m)
        self.sts[0] = self.s0 * self.u ** self.n
        for i in range(self.m)[1:]:
            self.sts[i] = self.sts[i - 1] * self.d

    def init_payoffs_tree(self):
        odd_nodes = self.sts[::2]
        if self.is_call:
            return np.maximum(0, odd_nodes - self.k)
        else:
            return np.maximum(0, self.k - odd_nodes)

    def check_early_exercise(self, payoffs, node):
        self.sts = self.sts[1:-1]
        odd_sts = self.sts[::2]
        if self.is_call:
            return np.maximum(payoffs, odd_sts - self.k)
        else:
            return np.maximum(payoffs, self.k - odd_sts)

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


class TrinomialLROptionLattice(StockOPtion, BasicTreeOptionsClass):
    def __init__(self, s0, k, r=0.05, t=1, n=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        super().__init__(s0, k, r, t, n, pu, pd, div, sigma, is_put, is_am)

    def setup_parameters(self):
        self.u = np.exp(self.sigma * np.sqrt(2. * self.dt))
        self.d = 1 / self.u
        self.m = 1
        self.qu = ((np.exp((self.r - self.div) * self.dt / 2.) -
                    np.exp(-self.sigma * np.sqrt(self.dt / 2.))) /
                   (np.exp(self.sigma * np.sqrt(self.dt / 2.)) - np.exp(
                       -self.sigma * np.sqrt(self.dt / 2.)))) ** 2
        self.qd = ((np.exp(self.sigma * np.sqrt(self.dt / 2.)) -
                    np.exp((self.r - self.div) * self.dt / 2.)) /
                   (np.exp(self.sigma * np.sqrt(self.dt / 2.)) -
                    np.exp(-self.sigma * np.sqrt(self.dt / 2.))
                    )) ** 2.

        self.qm = 1 - self.qu - self.qd
        self.m2 = 2 * self.n + 1

    def init_stock_price_tree(self):
        self.sts = np.zeros(self.m2)
        self.sts[0] = self.s0 * self.u ** self.n

        for i in range(self.m2)[1:]:
            self.sts[i] = self.sts[i - 1] * self.d


    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.sts - self.k)
        else:
            return np.maximum(0, self.k - self.sts)


    def check_early_exercise(self, payoffs, node):
        self.sts=self.sts[1:-1]
        if self.is_call:
            return np.maximum(payoffs, self.sts - self.k)
        else:
            return np.maximum(payoffs, self.k - self.sts)


    def traverse_tree(self, payoffs):
        for i in reversed(range(self.n)):
            payoffs = (payoffs[:-2] * self.qu + payoffs[1:-1] * self.qm + payoffs[2:] * self.qd) * self.df
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
    model = BinomialCRROption(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True)
    print(model.price())
    model = BinomialLROption(50, 52, r=0.05, t=2, n=4, sigma=0.3, is_put=True)
    print(model.price())
    model = BinomialLROption(50, 52, r=0.05, t=2, n=4, sigma=0.3, is_put=True, is_am=True)
    print(model.price())
    model = BinomialLROptionGreeks(50, 52, r=0.05, t=2, n=300, sigma=0.3)
    print(model.price())
    model = BinomialLROptionGreeks(50, 52, r=0.05, t=2, n=300, sigma=0.3, is_put=True)
    print(model.price())
    model = TrinomialLROption(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True, is_am=True)
    print(model.price())
    model = TrinomialLROption(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True)
    print(model.price())
    model = BinomialCRROptionLattice(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True)
    print(model.price())
    model = BinomialCRROptionLattice(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True, is_am=True)
    print(model.price())
    model = TrinomialLROptionLattice(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True)
    print(model.price())
    model = TrinomialLROptionLattice(50, 52, r=0.05, t=2, n=2, sigma=0.3, is_put=True, is_am=True)
    print(model.price())
