import sys

from vanilla_options_model.basic_class.abstract_basic import BasicFiniteDifferences
from vanilla_options_model.basic_class.finite_difference_basic_class import FiniteDifferenceBasicClass
import numpy as np
import scipy.linalg as linalg

import scipy.stats as st


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
                    new_values[k] = self.calculate_payoffs(k, rhs, old_values, new_values)

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


class VasicekCZCBOptions(object):
    def __init__(self):
        self.norminv = st.norm.ppf
        self.norm = st.norm.cdf

    def vasicke_czcb_values(self, r0, r, ratio, t, sigma, kappa, theta, m, prob=1e-6, max_policy_iter=10,
                            grid_struct_const=0.25, rs=None
                            ):
        (r_min, dr, n, dtau) = \
            self.vasicek_params(r0, m, sigma, kappa, theta, t, prob, grid_struct_const, rs)
        r = np.r_[0:n] * dr + r_min
        v_mplus1 = np.ones(n)
        for i in range(1, m + 1):
            k = self.exercise_call_price(r, ratio, i * dtau)
            eex = np.ones(n) * k
            (subdiagonal, diagonal, superdiagonal) = \
                self.vasicek_diagonals(
                    sigma, kappa, theta, r_min, dr, n, dtau
                )
            (v_mplus1, iterations) = \
                self.iterate(subdiagonal, diagonal, superdiagonal, v_mplus1, eex, max_policy_iter)
        return r, v_mplus1

    def vasicek_params(self, r0, m, sigma, kappa, theta, t, prob, grid_struct_const=0.25, rs=None):
        if rs is not None:
            (r_min, r_max) = (rs[0], rs[-1])
        else:
            (r_min, r_max) = self.vasicke_limits(r0, sigma, kappa, theta, t, prob)
        dt = t / float(m)
        n = self.calculate_n(grid_struct_const, dt, sigma, r_max, r_min)
        dr = (r_max - r_min) / (n - 1)
        return r_min, dr, n, dt

    def calculate_n(self, max_structure_const, dt, sigma, r_max, r_min):
        n = 0
        while 1:
            n += 1
            grid_structure_interval = \
                dt * (sigma ** 2) / (((r_max - r_min) / float(n) ** 2))
            if grid_structure_interval > max_structure_const:
                break
        return n

    def vasicke_limits(self, r0, sigma, kappa, theta, t, prob=1e-6):
        er = theta + (r0 - theta) * np.exp(-kappa * t)
        variance = (sigma ** 2) * t if kappa == 0 else \
            (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * t))
        stdev = np.sqrt(variance)
        r_min = self.norminv(prob, er, stdev)
        r_max = self.norminv(1 - prob, er, stdev)
        return r_min, r_max

    def vasicek_diagonals(self, sigma, kappa, theta, r_min, dr, n, dtau):
        rn = np.r_[0:n] * dr + r_min
        subdiagonals = kappa * (theta - rn) * dtau / (2 * dr) - \
                       0.5 * (sigma ** 2) * dtau / (dr ** 2)
        diagonals = 1 + rn * dtau + sigma ** 2 * dtau / (dr ** 2)
        superdiagnoals = -kappa * (theta - rn) * dtau / (2 * dr) - \
                         0.5 * (sigma ** 2) * dtau / (dr ** 2)
        if n > 0:
            v_subd0 = subdiagonals[0]
            superdiagnoals[0] = superdiagnoals[0] - subdiagonals[0]
            diagonals[0] += 2 * v_subd0
            subdiagonals[0] = 0

        if n > 1:
            v_subd0 = subdiagonals[-1]
            superdiagnoals[-1] = superdiagnoals[-1] - subdiagonals[-1]
            diagonals[-1] += 2 * v_subd0
            subdiagonals[-1] = 0

        return subdiagonals, diagonals, superdiagnoals

    def check_exercise(self, v, eex):
        return v > eex

    def exercise_call_price(self, r, ratio, tau):
        k = ratio * np.exp(-r * tau)
        return k

    def vasicek_policy_diagonals(self, subdiagonal, diagonal, superdiagonal, v_old, v_new, eex):
        has_early_exercise = self.check_exercise(v_new, eex)
        subdiagonal[has_early_exercise] = 0
        superdiagonal[has_early_exercise] = 0
        policy = v_old / eex
        policy_values = policy[has_early_exercise]
        diagonal[has_early_exercise] = policy_values
        return subdiagonal, diagonal, superdiagonal

    def iterate(self, subdiagonal, diagonal, superdiagonal, v_old, eex, max_policy_iter=10):
        v_mplus1 = v_old
        v_m = v_old
        change = np.zeros(len(v_old))
        prev_changes = np.zeros(len(v_old))

        iterations = 0
        while iterations <= max_policy_iter:
            iterations += 1
            v_mplus1 = self.tridiagonal_solve(subdiagonal, diagonal, superdiagonal, v_old)
            subdiagonal, diagonal, superdiagonal = \
                self.vasicek_policy_diagonals(subdiagonal, diagonal, superdiagonal, v_old, v_mplus1, eex)
            is_eex = self.check_exercise(v_mplus1, eex)
            change[is_eex] = 1

            if iterations > 1:
                change[v_mplus1 != v_m] = 1

            is_no_more_eex = False if True in is_eex else True
            if is_no_more_eex:
                break

            v_mplus1[is_eex] = eex[is_eex]
            changes = (change == prev_changes)

            is_no_further_changes = all((x == 1) for x in changes)
            if is_no_further_changes:
                break

            prev_changes = change
            v_m = v_mplus1
        return v_mplus1, iterations - 1

    def tridiagonal_solve(self, a, b, c, d):
        nf = len(a)
        ac, bc, cc, dc = map(np.array, (a, b, c, d))
        for it in range(1, nf):
            mc = ac[it] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]

        xc = ac
        xc[-1] = dc[-1] / bc[-1]

        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1] / bc[il])
        del bc, cc, dc

        return xc


if __name__ == '__main__':
    model = FDExplicitEU(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=1000, is_put=True)
    print(model.price())

    model = FDImplicitEU(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=1000, is_put=True)
    print(model.price())
    model = FDImplicitEU(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=80, n=100, is_put=True)
    print(model.price())

    model = FDCEu(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=1000, is_put=True)
    print(model.price())
    model = FDCEu(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=80, n=100, is_put=True)
    print(model.price())

    # model = FDCAM(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=100, n=42, omega=1.2, tol=0.001)
    # print(model.price())
    # model = FDCAM(50, 50, r=0.1, t=5. / 12., sigma=0.4, smax=100, m=80, n=100, omega=1.2, tol=0.001, is_put=True)
    # print(model.price())

    r0 = 0.05
    r = 0.05
    ratio = 0.95
    sigma = 0.03
    kappa = 0.15
    theta = 0.05
    prob = 1e-6
    m = 250
    max_policy_iter = 10
    grid_struct_interval = 0.25
    rs = np.r_[0.0:2.0:0.1]

    vasicek = VasicekCZCBOptions()
    r, vals = vasicek.vasicke_czcb_values(r0, r, ratio, 1., sigma, kappa, theta, m, prob, max_policy_iter,
                                          grid_struct_interval, rs)
    print(r)
    print('-'*100)
    print(vals)
    # https://conda.anaconda.org/nvidia/linux-64/cudnn-8.0.0-cuda11.0_0.tar.bz2

