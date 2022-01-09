import sys

from options_pricing_model_class.abstract_basic import BasicFiniteDifferences
from options_pricing_model_class.finite_difference_basic_class import FiniteDifferenceBasicClass
import numpy as np
import scipy.linalg as linalg


class FDExplicitEU(BasicFiniteDifferences, FiniteDifferenceBasicClass):
    def __init__(self, s0, k, r, t, sigma, smax, m, n, is_put=False):
        super().__init__(s0, k, r, t, sigma, smax, m, n, is_put)

    def setup_bounday_conditions(self):
        if self.is_call:
            self.grid[:, -1] = np.maximum(
                0, self.boundary_conds - self.k
            )
            self.grid[-1, :-1] = (self.smax - self.k) * np.exp(-self.r * self.dt * (self.n - self.j_values))
        else:
            self.grid[:, -1] = np.maximum(0, self.k - self.boundary_conds)
            self.grid[0, :-1] = (self.k - self.smax) * np.exp(-self.r * self.dt * (self.n - self.j_values))

    def setup_coefficients(self):
        self.a = 0.5 * self.dt * ((self.sigma ** 2) * (self.i_values ** 2) - self.r * self.i_values)
        self.b = 1 - self.dt * ((self.sigma ** 2) * (self.i_values ** 2) + self.r)
        self.c = 0.5 * self.dt * ((self.sigma ** 2) * (self.i_values ** 2) + self.r * self.i_values)

    def traverse_grid(self):
        for j in reversed(self.j_values):
            for i in range(self.m)[2:]:
                self.grid[i, j] = self.a[i] * self.grid[i - 1, j + 1] + self.b[i] * self.grid[i, j + 1] + self.c[i] * \
                                  self.grid[i + 1, j + 1]

    def interpolate(self):
        return np.interp(self.s0, self.boundary_conds, self.grid[:, 0])

    def price(self):
        self.setup_bounday_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()


class FDImplicitEU(BasicFiniteDifferences, FiniteDifferenceBasicClass):
    def __init__(self, s0, k, r, t, sigma, smax, m, n, is_put=False):
        super().__init__(s0, k, r, t, sigma, smax, m, n, is_put)

    def setup_bounday_conditions(self):
        if self.is_call:
            self.grid[:, -1] = np.maximum(
                0, self.boundary_conds - self.k
            )
            self.grid[-1, :-1] = (self.smax - self.k) * np.exp(-self.r * self.dt * (self.n - self.j_values))
        else:
            self.grid[:, -1] = np.maximum(0, self.k - self.boundary_conds)
            self.grid[0, :-1] = (self.k - self.smax) * np.exp(-self.r * self.dt * (self.n - self.j_values))

    def setup_coefficients(self):
        self.a = 0.5 * (self.r * self.dt * self.i_values - (self.sigma ** 2) * self.dt * (self.i_values ** 2))
        self.b = 1 + (self.sigma ** 2) * self.dt * (self.i_values ** 2) + self.r * self.dt
        self.c = -0.5 * (self.r * self.dt * self.i_values + (self.sigma ** 2) * self.dt * (self.i_values ** 2))
        self.coeffs = np.diag(self.a[2:self.m], -1) + np.diag(self.b[1:self.m]) + np.diag(self.c[1:self.m - 1], 1)

    def traverse_grid(self):
        p, l, u = linalg.lu(self.coeffs)
        aux = np.zeros(self.m - 1)
        for j in reversed(range(self.n)):
            aux[0] = np.dot(-self.a[1], self.grid[0, j])
            x1 = linalg.solve(l, self.grid[1:self.m, j + 1] + aux)
            x2 = linalg.solve(u, x1)
            self.grid[1:self.m, j] = x2

    def interpolate(self):
        return np.interp(self.s0, self.boundary_conds, self.grid[:, 0])

    def price(self):
        self.setup_bounday_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()


class FDCEu(BasicFiniteDifferences, FiniteDifferenceBasicClass):
    def __init__(self, s0, k, r, t, sigma, smax, m, n, is_put=False):
        super().__init__(s0, k, r, t, sigma, smax, m, n, is_put)

    def setup_bounday_conditions(self):
        if self.is_call:
            self.grid[:, -1] = np.maximum(
                0, self.boundary_conds - self.k
            )
            self.grid[-1, :-1] = (self.smax - self.k) * np.exp(-self.r * self.dt * (self.n - self.j_values))
        else:
            self.grid[:, -1] = np.maximum(0, self.k - self.boundary_conds)
            self.grid[0, :-1] = (self.k - self.smax) * np.exp(-self.r * self.dt * (self.n - self.j_values))

    def setup_coefficients(self):
        self.alpha = 0.25 * self.dt * ((self.sigma ** 2) * (self.i_values ** 2) - self.r * self.i_values)
        self.beta = -self.dt * 0.5 * ((self.sigma ** 2) * (self.i_values ** 2) + self.r)
        self.gamma = 0.25 * self.dt * ((self.sigma ** 2) * (self.i_values ** 2) + self.r * self.i_values)
        self.m1 = -np.diag(self.alpha[2:self.m], -1) + np.diag(1 - self.beta[1:self.m]) - np.diag(
            self.gamma[1:self.m - 1], 1)
        self.m2 = np.diag(self.alpha[2:self.m], -1) + np.diag(1 + self.beta[1:self.m]) + np.diag(
            self.gamma[1:self.m - 1], 1)

    def traverse_grid(self):
        p, l, u = linalg.lu(self.m1)
        for j in reversed(range(self.n)):
            x1 = linalg.solve(l, np.dot(self.m2, self.grid[1:self.m, j + 1]))
            x2 = linalg.solve(u, x1)
            self.grid[1:self.m, j] = x2

    def interpolate(self):
        return np.interp(self.s0, self.boundary_conds, self.grid[:, 0])

    def price(self):
        self.setup_bounday_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()


class FDCAM(BasicFiniteDifferences, FiniteDifferenceBasicClass):
    def __init__(self, s0, k, r, t, sigma, smax, m, n, tol, omega, is_put=False):
        super().__init__(s0, k, r, t, sigma, smax, m, n, is_put)
        self.omega = omega
        self.tol = tol
        self.i_values = np.arange(self.m + 1)
        self.j_values = np.arange(self.n + 1)

    def setup_bounday_conditions(self):
        if self.is_call:
            self.payoffs = np.maximum(0, self.boundary_conds[1:self.m] - self.k)
        else:
            self.payoffs = np.maximum(0, self.k - self.boundary_conds[1:self.m])

        self.past_values = self.payoffs
        self.boundary_values = self.k * np.exp(-self.r * self.dt * (self.n - self.j_values))

    def setup_coefficients(self):
        self.alpha = 0.25 * self.dt * ((self.sigma ** 2) * (self.i_values ** 2) - self.r * self.i_values)
        self.beta = -self.dt * 0.5 * ((self.sigma ** 2) * (self.i_values ** 2) + self.r)
        self.gamma = 0.25 * self.dt * ((self.sigma ** 2) * (self.i_values ** 2) + self.r * self.i_values)
        self.m1 = -np.diag(self.alpha[2:self.m], -1) + np.diag(1 - self.beta[1:self.m]) - np.diag(
            self.gamma[1:self.m - 1], 1)
        self.m2 = np.diag(self.alpha[2:self.m], -1) + np.diag(1 + self.beta[1:self.m]) + np.diag(
            self.gamma[1:self.m - 1], 1)

    def traverse_grid(self):
        aux = np.zeros(self.m - 1)
        new_values = np.zeros(self.m - 1)
        for j in reversed(range(self.n)):
            aux[0] = self.alpha[1] * (self.boundary_values[j] + self.boundary_values[j + 1])
            rhs = np.dot(self.m2, self.past_values) + aux
            old_values = np.copy(self.past_values)
            error = sys.float_info.max

            while self.tol < error:
                new_values[0] = self.calculate_payoff_start_boundary(rhs, old_values)
                for k in range(self.m - 2)[1:]:
                    new_values[k] = self.calculate_payoffs(k, rhs,old_values, new_values)

                new_values[-1] = self.calculate_payoff_end_boundary(rhs, old_values, new_values)

                error = np.linalg.norm(new_values - old_values)
                old_values = np.copy(new_values)

        self.past_values = np.copy(new_values)

        self.values = np.concatenate(([self.boundary_values[0]], new_values, 0))

    def calculate_payoff_start_boundary(self, rhs, old_values):
        payoff = old_values[0] + self.omega / (1 - self.beta[1]) * (
                rhs[0] - (1 - self.beta[1] * old_values[0] + self.gamma[1] * old_values[1]))
        return max(self.payoffs[0], payoff)

    def calculate_payoff_end_boundary(self, rhs, old_values, new_values):
        payoff = old_values[-1] + self.omega / (1 - self.beta[-2]) * (
                rhs[-1] + self.alpha[-2] * new_values[-2] - (1 - self.beta[-2]) * old_values[-1])
        return max(self.payoffs[-1], payoff)

    def calculate_payoffs(self, k, rhs, old_values, new_values):
        payoff = old_values[k] + self.omega / (1 - self.beta[k + 1]) * (
                rhs[k] + self.alpha[k + 1] * new_values[k - 1] - (1 - self.beta[k + 1]) * old_values[k] +
                self.gamma[k + 1] * old_values[k + 1])
        return max(self.payoffs[k], payoff)

    def interpolate(self):
        return np.interp(self.s0, self.boundary_conds, self.values)

    def price(self):
        self.setup_bounday_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()


if __name__ == '__main__':
    # model = FDExplicitEU(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=1000, is_put=True)
    # print(model.price())
    #
    # model = FDImplicitEU(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=1000, is_put=True)
    # print(model.price())
    # model = FDImplicitEU(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=80, n=100, is_put=True)
    # print(model.price())
    #
    # model = FDCEu(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=1000, is_put=True)
    # print(model.price())
    # model = FDCEu(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=80, n=100, is_put=True)
    # print(model.price())

    model = FDCAM(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=42, omega=1.2, tol=0.001)
    print(model.price())
    model = FDCAM(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=80, n=100, omega=1.2, tol=0.001, is_put=True)
    print(model.price())
