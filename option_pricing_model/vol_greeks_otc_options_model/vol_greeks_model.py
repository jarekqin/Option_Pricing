import numpy as np
from scipy.stats import norm
import pandas as pd
import math


def implied_volatility(options_price, underlying_price, strike_price, risk_free, t0, t1, options_type):
    """
    calculate implied volatility with bional method
    :param options_price: options price
    :param underlying_price: underlying price
    :param strike_price: strike price on options
    :param risk_free: risk free rate
    :param t0: time 0
    :param t1: time 1
    :param options_type: options_type
    :return: implied volatility
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    t = (t1 - t0).days / 365
    sigma_min = 1e-5
    sigma_max = 1.
    sigma_mid = (sigma_min + sigma_max) / 2

    if options_type in ['call', 'CALL', 'Call']:
        def call_bs(s, k, sigma, r, t):
            d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            call_value = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
            return call_value

        call_min = call_bs(underlying_price, strike_price, sigma_min, risk_free, t)
        call_max = call_bs(underlying_price, strike_price, sigma_max, risk_free, t)
        call_mid = call_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
        diff = options_price - call_mid
        if options_price < call_min or options_price > call_max:
            raise ValueError('options price error!')
        while abs(diff) > 1e-6:
            diff = options_price - call_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            sigma_mid = (sigma_min + sigma_max) / 2
            call_mid = call_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            if options_price > call_mid:
                sigma_min = sigma_mid
            else:
                sigma_max = sigma_mid
    else:
        def put_bs(s, k, sigma, r, t):
            d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            put_value = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
            return put_value

        put_min = put_bs(underlying_price, strike_price, sigma_min, risk_free, t)
        put_max = put_bs(underlying_price, strike_price, sigma_max, risk_free, t)
        put_mid = put_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
        diff = options_price - put_mid
        if options_price < put_min or options_price > put_max:
            raise ValueError('options price error!')
        while abs(diff) > 1e-6:
            diff = options_price - put_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            sigma_mid = (sigma_min + sigma_max) / 2
            put_mid = put_bs(underlying_price, strike_price, sigma_mid, risk_free, t)
            if options_price > put_mid:
                sigma_min = sigma_mid
            else:
                sigma_max = sigma_mid
    return sigma_mid


def BSM(s, k, sigma, r, t0, t1, types):
    """
    calculate options price with BSM model
    :param s: underlying price
    :param k: strike price
    :param sigma: underlying returns volatility
    :param r: risk free rate
    :param t0: time start
    :param t1: time to maturity
    :param types: options type
    :return: options price
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    t = (t1 - t0).days / 365
    d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if types in ['call', 'CALL', 'Call']:
        options_price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    elif types in ['put', 'PUT', 'Put']:
        options_price = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    else:
        raise TypeError('types only supports "call/put"!')
    return options_price


def eu_options_letters(s, k, sigma, r, t0, t1, letter, types):
    """
    calculate eu options price with BSM model
    :param s: underlying price
    :param k: strike price
    :param sigma: underlying returns volatility
    :param r: risk free rate
    :param t0: time start
    :param t1: time to maturity
    :param letter: letter type
    :param types: options type
    :return: options letter value
    """
    t0, t1 = pd.to_datetime(t0).date(), pd.to_datetime(t1).date()
    t = (t1 - t0).days / 365
    d1 = (np.log(s / k) + (r + pow(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    result = None

    if letter in ['Delta', 'delta', 'DELTA']:
        if types in ['call', 'CALL', 'Call']:
            result = norm.cdf(d1)
        elif types in ['put', 'PUT', 'Put']:
            result = norm.cdf(d1) - 1
        else:
            raise TypeError('types only supports "call/put"!')

    elif letter in ['Gamma', 'gamma', 'GAMMA']:
        result = np.exp(-pow(d1, 2) / 2) / (s * sigma * np.sqrt(2 * np.pi * t))

    elif letter in ['Theta', 'theta', 'THETA']:
        if types in ['call', 'CALL', 'Call']:
            result = -(s * sigma * np.exp(-pow(d1, 2) / 2)) / (2 * np.sqrt(2 * np.pi * t)) - r * k * np.exp(
                (-r * t) * norm.cdf(d2))
        elif types in ['put', 'PUT', 'Put']:
            result = -(s * sigma * np.exp(-pow(d1, 2) / 2)) / (2 * np.sqrt(2 * np.pi * t)) + r * k * np.exp(
                (-r * t) * norm.cdf(-d2))

    elif letter in ['Vega', 'vega', 'VEGA']:
        result = s * np.sqrt(t) * np.exp(-pow(d1, 2) / 2) / np.sqrt(2 * np.pi)

    elif letter in ['Rho', 'rho', 'RHO']:
        if types in ['call', 'CALL', 'Call']:
            result = k * t * np.exp(-r * t) * norm.cdf(d2)
        elif types in ['put', 'PUT', 'Put']:
            result = -k * t * np.exp(-r * t) * norm.cdf(-d2)
        else:
            raise TypeError('types only supports "call/put"!')

    else:
        raise TypeError('wrong type of options letter!')

    return result


def best_hedged_ratio(hedge_ratio, asset_price, futures_contract_fv):
    """
    calculate best hedged ratio between underlying and futures
    :param hedge_ratio: ratio of hedge
    :param asset_price: underlying asset
    :param futures_contract_fv: futures fv
    :return: best ratio
    """
    N = hedge_ratio * asset_price / futures_contract_fv
    if math.modf(N)[0] > 0.5:
        N = math.ceil(N)
    else:
        N = math.floor(N)
    return N


def options_parity(call_price, put_price, underlying_price, strike_price, risk_free, maturity, option_type):
    """
    calculate options parity
    :param call_price: call options price
    :param put_price: put options price
    :param underlying_price: underlying price
    :param strike_price: strike price on options
    :param risk_free: risk free rate
    :param maturity: maturity of options
    :param option_type: options type only supports "call"/"put" lower letter
    :return: equitum value
    """
    if option_type == 'call':
        return put_price + underlying_price - strike_price * np.exp(-risk_free * maturity)
    else:
        return call_price + strike_price * np.exp(-risk_free * maturity) - underlying_price


def binary_tree_eu_options_cal(underlying_price, strike_price, sigma, risk_free, maturity, steps, options_type):
    """
    binary tree eu options calculator
    :param underlying_price: underlying price
    :param strike_price: strike price on options
    :param sigma: volatility on underlying
    :param risk_free: risk free rate
    :param maturity: maturity on options
    :param steps: simulation steps
    :param options_type: options type only supports "call"/"put"
    :return: options value
    """
    from math import factorial
    t = maturity / steps
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(risk_free * t) - d) / (u - d)
    n_list = range(0, steps + 1)
    A = []
    for i in n_list:
        c_nj = np.maximum(underlying_price * pow(u, i) * pow(d, steps - i) - strike_price, 0)
        num = factorial(steps) / (factorial(i) * factorial(steps - i))
        A.append(num * pow(p, i) * pow(1 - p, steps - i) * c_nj)
    if options_type == 'call':
        return np.exp(-risk_free * maturity) * np.sum(A)
    elif options_type == 'put':
        return np.exp(-risk_free * maturity) * np.sum(A) + strike_price * np.exp(
            -risk_free * maturity) - underlying_price
    else:
        raise TypeError('options_type only supports "call/put" with lower letters!')


def binary_tree_us_options_cal_and_letters(underlying_price, strike_price, sigma, risk_free, maturity, steps,
                                           options_type,
                                           return_letters=True, position='long'):
    """
    binary tree eu options calculator
    :param underlying_price: underlying price
    :param strike_price: strike price on options
    :param sigma: volatility on underlying
    :param risk_free: risk free rate
    :param maturity: maturity on options
    :param steps: simulation steps
    :param options_type: options type only supports "call"/"put"
    :param return_letters: whether return Greek letters
    :param position: holding options direction
    :return: options value
    """
    from math import factorial
    t = maturity / steps
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(risk_free * t) - d) / (u - d)
    options_matrix = np.zeros((steps + 1, steps + 1))
    n_list = np.arange(0, steps + 1)
    s_end = underlying_price * pow(u, steps - n_list) * pow(d, n_list)
    if options_type == 'call':
        options_matrix[:, -1] = np.maximum(s_end - strike_price, 0)
        i_list = list(range(0, steps))
        i_list.reverse()
        for i in i_list:
            j_list = np.arange(i + 1)
            si = underlying_price * pow(u, i - j_list) * pow(d, j_list)
            call_strike = np.maximum(si - strike_price, 0)
            call_nostrike = (p * options_matrix[:i + 1, i + 1] + (1 - p) * options_matrix[1:i + 2, i + 1]) * \
                            np.exp(-risk_free * t)
            options_matrix[:i + 1, i] = np.maximum(call_strike, call_nostrike)
        if return_letters:
            greek = {}
            delta_ = (options_matrix[0, 1] - options_matrix[1, 1]) / (underlying_price * u - underlying_price * d)
            gamma_delta_1 = (options_matrix[0, 2] - options_matrix[1, 2]) / (
                    underlying_price * pow(u, 2) - underlying_price)
            gamma_delta_2 = (options_matrix[1, 2] - options_matrix[2, 2]) / (
                    underlying_price - underlying_price * pow(d, 2))
            gamma_ = 2 * (gamma_delta_1 - gamma_delta_2) / (underlying_price * pow(u, 2) - underlying_price * pow(d, 2))
            theta_ = (options_matrix[1, 2] - options_matrix[0, 0]) / (2 * t)
            vega_ = (binary_tree_us_options_cal_and_letters(underlying_price, strike_price, sigma + 1e-4,
                                                risk_free,
                                                maturity, steps, options_type,
                                                False, position) - options_matrix[0, 0]) / 1e-4
            rho_ = (binary_tree_us_options_cal_and_letters(underlying_price, strike_price, sigma,
                                               risk_free + 1e-4,
                                               maturity, steps, options_type,
                                               False, position) - options_matrix[0, 0]) / 1e-4
            if position == 'long':
                greek['delta'] = delta_
            elif position == 'short':
                greek['delta'] = -delta_
            else:
                raise TypeError('wrong position value,only supporting "long/short"!')
            greek['gamma'] = gamma_
            greek['theta'] = theta_
            greek['vega'] = vega_
            greek['rho'] = rho_
            return options_matrix[0, 0], greek
        else:
            return options_matrix[0, 0]
    elif options_type == 'put':
        options_matrix[:, -1] = np.maximum(strike_price - s_end, 0)
        i_list = list(range(0, steps))
        i_list.reverse()
        for i in i_list:
            j_list = np.arange(i + 1)
            si = underlying_price * pow(u, i - j_list) * pow(d, j_list)
            put_strike = np.maximum(strike_price - si, 0)
            put_nostrike = np.exp(-risk_free * t) * (
                    p * options_matrix[:i + 1, i + 1] + (1 - p) * options_matrix[1:i + 2, i + 1])
            options_matrix[:i + 1, i] = np.maximum(put_strike, put_nostrike)
        if return_letters:
            greek = {}
            delta_ = (options_matrix[0, 1] - options_matrix[1, 1]) / (underlying_price * u - underlying_price * d)
            gamma_delta_1 = (options_matrix[0, 2] - options_matrix[1, 2]) / (
                    underlying_price * pow(u, 2) - underlying_price)
            gamma_delta_2 = (options_matrix[1, 2] - options_matrix[2, 2]) / (
                    underlying_price - underlying_price * pow(d, 2))
            gamma_ = 2 * (gamma_delta_1 - gamma_delta_2) / (underlying_price * pow(u, 2) - underlying_price * pow(d, 2))
            theta_ = (options_matrix[1, 2] - options_matrix[0, 0]) / (2 * t)
            vega_ = (binary_tree_us_options_cal_and_letters(underlying_price, strike_price, sigma + 1e-4,
                                                risk_free,
                                                maturity, steps, options_type,
                                                False, position) - options_matrix[0, 0]) / 1e-4
            rho_ = (binary_tree_us_options_cal_and_letters(underlying_price, strike_price, sigma,
                                               risk_free + 1e-4,
                                               maturity, steps, options_type,
                                               False, position) - options_matrix[0, 0]) / 1e-4
            if position == 'long':
                greek['delta'] = delta_
            elif position == 'short':
                greek['delta'] = -delta_
            else:
                raise TypeError('wrong position value,only supporting "long/short"!')
            greek['gamma'] = gamma_
            greek['theta'] = theta_
            greek['vega'] = vega_
            greek['rho'] = rho_
            return options_matrix[0, 0], greek
        else:
            return options_matrix[0, 0]


if __name__ == '__main__':
    print('call options price: ', BSM(325, 350, 0.291239, 0.0231393, '2019-10-02', '2020-04-02', 'call'))
    print('put options price: ', BSM(325, 310, 0.291239, 0.0233875, '2019-10-02', '2020-10-02', 'put'))
    print('*' * 50, 'beautiful line', '*' * 50)
    t_cal = '2019-06-28'
    t_maturity_call = '2019-08-28'
    t_maturity_put = '2019-09-25'
    vol = 0.234811
    underlying_price = 2.950

    k_call = 3.2
    k_put = 2.8
    risk_free = 0.02708

    print(eu_options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'delta', 'call'))
    print(eu_options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'gamma', 'call'))
    print(eu_options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'vega', 'call'))
    print(eu_options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'theta', 'call'))
    print(eu_options_letters(underlying_price, k_call, vol, risk_free, t_cal, t_maturity_call, 'rho', 'call'))

    print('*' * 50, 'beautiful line', '*' * 50)
    print(eu_options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'delta', 'put'))
    print(eu_options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'gamma', 'put'))
    print(eu_options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'vega', 'put'))
    print(eu_options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'theta', 'put'))
    print(eu_options_letters(underlying_price, k_put, vol, risk_free, t_cal, t_maturity_put, 'rho', 'put'))

    print('*' * 50, 'beautiful line', '*' * 50)
    print(best_hedged_ratio(1, 22650000, 300 * 1000))

    print('*' * 50, 'beautiful line', '*' * 50)
    from datetime import datetime

    p_etf_apr25 = 2.913
    shibor_apr25 = 0.0288
    t_calculate = datetime(2019, 4, 25)
    t_mature = datetime(2019, 12, 25)

    k_c28 = 2.8
    k_c30 = 3.0
    k_c32 = 3.2

    p_c28_apr25 = 0.3432
    p_c30 = apr25 = 0.2438
    p_c32_apr25 = 0.1688

    print(implied_volatility(p_c28_apr25, p_etf_apr25, k_c28, shibor_apr25, t_calculate, t_mature, 'call'))
    print('*' * 50, 'beautiful line', '*' * 50)
    print(options_parity(0.15, 0.3, 5.0, 5.2, 0.02601, 3 / 12, "call"))
    print('*' * 50, 'beautiful line', '*' * 50)
    print(binary_tree_eu_options_cal(6.32, 6.6, 0.2538, 0.0228, 1, 250, 'call'))
    print('*' * 50, 'beautiful line', '*' * 50)
    print(binary_tree_us_options_cal_and_letters(3.5, 3.8, 0.1676, 0.02, 1, 252, 'put'))
    print(binary_tree_us_options_cal_and_letters(3.5, 3.8, 0.1676, 0.02, 1, 252, 'call'))
    print('*' * 50, 'beautiful line', '*' * 50)
    print(binary_tree_us_options_cal_and_letters(3.27, 3.6, 0.19, 0.02377, 0.5, 100, 'call', 'long'))
    print(binary_tree_us_options_cal_and_letters(3.27, 3.6, 0.19, 0.02377, 0.5, 100, 'put', 'short'))
