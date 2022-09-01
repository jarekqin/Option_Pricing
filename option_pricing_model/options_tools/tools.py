import numpy as np
from scipy.stats import norm

from option_pricing_model.vanilla_options_model.bsm_base import BSM, BSM_DELTA


def CBND(a, b, rho):
    """
    cumulative binominal normal distribution
    :param a: params 1
    :param b: params 2
    :param rho: correlation between a and b
    :return:
    """
    x = [0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334]
    y = [0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604]
    a1 = a / np.sqrt(2 * (1 - rho ** 2))
    b1 = b / np.sqrt(2 * (1 - rho ** 2))

    if a <= 0 and b <= 0 and rho <= 0:
        result = 0
        for i in range(5):
            for j in range(5):
                result = result + x[i] * x[j] * np.exp(a1 * (2 * y[i] - a1) + \
                                                       b1 * (2 * y[j] - b1) + 2 * rho * (y[i] - a1) * (y[j] - b1))
        return np.sqrt(1 - rho ** 2) / np.pi * result
    elif a <= 0 and b >= 0 and rho >= 0:
        return norm.cdf(a) - CBND(a, -b, -rho)
    elif a >= 0 and b <= 0 and rho >= 0:
        return norm.cdf(b) - CBND(-a, b, -rho)
    elif a >= 0 and b >= 0 and rho <= 0:
        return norm.cdf(a) + norm.cdf(b) - 1 + CBND(-a, -b, rho)
    elif a * b * rho > 0:
        rho1 = (rho * a - b) * np.sign(a) / np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
        rho2 = (rho * b - a) * np.sign(b) / np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
        delta = (1 - np.sign(a) * np.sign(b)) / 4
        return CBND(a, 0, rho1) + CBND(b, 0, rho2) - delta
    else:
        raise ValueError


def Kc(strike, maturity, rate, carry_cost, vol):
    n = 2 * carry_cost / vol ** 2
    m = 2 * rate / vol ** 2
    q2u = (-(n - 1) + np.sqrt((n - 1) ** 2 + 4 * m)) / 2
    su = strike / (1 - 1 / q2u)
    h2 = -(carry_cost * maturity + 2 * vol * np.sqrt(maturity)) * strike / (su - strike)
    si = strike + (su - strike) * (1 - np.exp(h2))

    k = 2 * rate / (vol ** 2 * (1 - np.exp(-rate * maturity)))
    d1 = (np.log(si / strike) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
    q2 = (-(n - 1) + np.sqrt((n - 1) ** 2 + 4 * k)) / 2
    lhs = si - strike
    rhs = BSM(si, strike, maturity, rate, carry_cost, vol) + (
            1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(d1)) * si / q2
    bi = np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) * (1 - 1 / q2) + (
            1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) / (vol * np.sqrt(maturity))) / q2
    e = 1e-6

    while abs(lhs - rhs) / strike > e:
        si = (strike + rhs - bi * si) / (1 - bi)
        d1 = (np.log(si / strike) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
        lhs = si - strike
        rhs = BSM(si, strike, maturity, rate, carry_cost, vol) + (
                1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(d1)) * si / q2
        bi = np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) * (1 - 1 / q2) + (
                1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) / (vol * np.sqrt(maturity))) / q2
    return si


def kp(strike, maturity, rate, carry_cost, vol):
    n = 2 * carry_cost / vol ** 2
    m = 2 * rate / vol ** 2
    qlu = (-(n - 1) - np.sqrt((n - 1) ** 2 + 4 * m)) / 2
    su = strike / (1 - 1 / qlu)
    h1 = (carry_cost * maturity - 2 * vol * np.sqrt(maturity)) * strike / (strike - su)
    si = su + (strike - su) * np.exp(h1)

    k = 2 * rate / (vol ** 2 * (1 - np.exp(-rate * maturity)))
    d1 = (np.log(si / strike) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
    q1 = (-(n - 1) - np.sqrt((n - 1) ** 2 + 4 * k)) / 2
    lhs = strike - si
    rhs = BSM(si, strike, maturity, rate, carry_cost, vol, 'put') - (
            1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1)) * si / q1
    bi = -np.exp((carry_cost - rate) * maturity) + norm.cdf(-d1) * (1 - 1 / q1) - (
            1 + np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1) / (vol * np.sqrt(maturity))) / q1
    e = 1e-6

    while abs(lhs - rhs) / strike > e:
        si = (strike - rhs + bi * si) / (1 + bi)
        d1 = (np.log(si / strike) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
        lhs = strike - si
        rhs = BSM(si, strike, maturity, rate, carry_cost, vol, 'put') - (
                1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1)) * si / q1
        bi = -np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1) * (1 - 1 / q1) - (
                1 + np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1) / (vol * np.sqrt(maturity))) / q1
    return si


def phi(underlying_price, maturity, gamma, H, I, rate, carry_cost, vol):
    lambda1 = (-rate + gamma * carry_cost + 0.5 * gamma * (gamma - 1) * vol ** 2) * maturity
    d = -(np.log(underlying_price / H) + (carry_cost + (gamma - 0.5) * vol ** 2) * maturity) / (vol * np.sqrt(maturity))
    kappa = 2 * carry_cost / (vol ** 2) + (2 * gamma - 1)
    return np.exp(lambda1) * underlying_price ** gamma * (norm.cdf(d) - (I / underlying_price) ** kappa * norm.cdf(
        d - 2 * np.log(I / underlying_price) / (vol * np.sqrt(maturity))))


def CriticalValueChooser(underlying_price, strike_price_call, strike_price_put, chooser_time, maturity_call,
                         maturity_put, rate, carry_cost, vol):
    Sv = underlying_price
    ci = BSM(Sv, strike_price_call, maturity_call - chooser_time, rate, carry_cost, vol)
    Pi = BSM(Sv, strike_price_put, maturity_put - chooser_time, rate, carry_cost, vol, 'put')
    dc = BSM_DELTA(Sv, strike_price_call, maturity_call - chooser_time, rate, carry_cost, vol)
    dp = BSM_DELTA(Sv, strike_price_put, maturity_put - chooser_time, rate, carry_cost, vol, 'put')
    yi = ci - Pi
    di = dc - dp
    epsilon = 0.001
    while abs(yi) > epsilon:
        Sv = Sv - (yi) / di
        ci = BSM(Sv, strike_price_call, maturity_call - chooser_time, rate, carry_cost, vol)
        Pi = BSM(Sv, strike_price_put, maturity_put - chooser_time, rate, carry_cost, vol, 'put')
        dc = BSM_DELTA(Sv, strike_price_call, maturity_call - chooser_time, rate, carry_cost, vol)
        dp = BSM_DELTA(Sv, strike_price_put, maturity_put - chooser_time, rate, carry_cost, vol, 'put')
        yi = ci - Pi
        di = dc - dp

    return Sv


def CriticalValueOptionsOnOptions(strike_price1, strike_price2, maturity, rate, carray_cost, vol, options_type):
    Si = strike_price1
    ci = BSM(Si, strike_price1, maturity, rate, carray_cost, vol, options_type)
    di = BSM_DELTA(Si, strike_price1, maturity, rate, carray_cost, vol, options_type)
    epsilon = 0.000001

    while abs(ci - strike_price2) > epsilon:
        Si = Si - (ci - strike_price2) / di
        ci = BSM(Si, strike_price1, maturity, rate, carray_cost, vol, options_type)
        di = BSM_DELTA(Si, strike_price1, maturity, rate, carray_cost, vol, options_type)

    return Si


def CriticalPrice(id_, I1, t1, t2, v, q):
    Ii = I1
    yi = CriticalPart3(id_, Ii, t1, t2, v)
    di = CriticalPart2(id_, Ii, t1, t2, v)
    epsilon = 1e-5
    while abs(yi - q) > epsilon:
        Ii = Ii - (yi - q) / di
        yi = CriticalPart3(id_, Ii, t1, t2, v)
        di = CriticalPart2(id_, Ii, t1, t2, v)
    return Ii


def CriticalPart2(id_, I, t1, t2, v):
    if id_ == 1:
        z1 = (np.log(I) + v ** 2 / 2 * (t2 - t1)) / (v * np.sqrt(t2 - t1))
        return norm.cdf(z1)
    elif id_ == 2:
        z2 = (-np.log(I) - v ** 2 / 2 * (t2 - t1)) / (v * np.sqrt(t2 - t1))
        return -norm.cdf(z2)
    else:
        raise NotImplemented

def CriticalPart3(id_, I, t1, t2, v):
    if id_ == 1:
        z1 = (np.log(I) + v ** 2 / 2 * (t2 - t1)) / (v * np.sqrt(t2 - t1))
        z2 = (np.log(I) - v ** 2 / 2 * (t2 - t1)) / (v * np.sqrt(t2 - t1))
        return I * norm.cdf(z1) - norm.cdf(z2)
    elif id_ == 2:
        z1 = (-np.log(I) + v ** 2 / 2 * (t2 - t1)) / (v * np.sqrt(t2 - t1))
        z2 = (-np.log(I) - v ** 2 / 2 * (t2 - t1)) / (v * np.sqrt(t2 - t1))
        return norm.cdf(z1) - I * norm.cdf(z2)
    else:
        pass