import numpy as np
from scipy.stats import norm


def pd_merton(stock_value, debt_value, company_value, sigma, risk_free, maturity):
    """
    calculate probability of default on merton model
    :param stock_value: stock value
    :param debt_value: debt value
    :param company_value: company value
    :param risk_free: risk free rate
    :param maturity: maturity of debt
    :return: probability
    """
    d1 = (np.log(company_value / debt_value) + (risk_free + pow(sigma, 2) / 2 * maturity)) / (
            sigma * np.sqrt(maturity)
    )
    d2 = d1 - sigma * np.sqrt(maturity)
    return norm.cdf(-d2)


def convert_bond_binal_tree(stock_price, sigma, principle, convert_ratio, PD, risk_free, recovery_rate,
                            return_price, maturity, steps
                            ):
    """
    calculate convert bond options price with binary tree method
    :param stock_price: underlying stock price
    :param sigma: volatility of stock
    :param principle: principle on debt
    :param convert_ratio: convert ratio
    :param pd: probability of default
    :param risk_free: risk free rate
    :param recovery_rate: recovery rate when default
    :param return_price: selling back price
    :param maturity: bond maturity
    :param steps: tree steps
    :return: options price
    """
    t = maturity / steps
    u = np.exp(np.sqrt((pow(sigma, 2) - PD) * t))
    d = 1 / u
    pu = (np.exp(risk_free * t) - d * np.exp(-PD * t)) / (u - d)
    pd = (u * np.exp(-PD * t) - np.exp(risk_free * t)) / (u - d)
    p_default = 1 - np.exp(-PD * t)
    d_value = principle * recovery_rate
    cb_matrix = np.zeros((steps + 1, steps + 1))
    n_list = np.arange(0, steps + 1)
    s_end = stock_price * pow(u, steps - n_list) * pow(d, n_list)
    Q1 = principle
    Q3 = convert_ratio * s_end
    cb_matrix[:, -1] = np.maximum(np.minimum(Q1, return_price), Q3)
    i_list = list(range(0, steps))
    i_list.reverse()
    for i in i_list:
        j_list = np.arange(i + 1)
        si = stock_price * pow(u, i - j_list) * pow(d, j_list)
        Q1 = np.exp(-risk_free * t) * (
                pu * cb_matrix[:i + 1, i + 1] + pd * cb_matrix[1:i + 2, i + 1] + p_default * d_value)
        Q3 = convert_ratio * si
        cb_matrix[:i + 1, i] = np.maximum(np.minimum(Q1, return_price), Q3)
    return cb_matrix[0, 0]


def bsm_futures_options(futures_price, strike_price, sigma, risk_free, maturity, options_type):
    """
    calculate futures options
    :param futures_price: futures price
    :param strike_price: futures strike price
    :param sigma: volalitility of futures
    :param risk_free: risk free rate
    :param maturity: maturity of futures contrace
    :param options_type: options type
    :return: options price
    """
    d1 = (np.log(futures_price / strike_price) + pow(sigma, 2) * maturity / 2) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    if options_type in ['call', 'CALL', 'Call']:
        return np.exp(-risk_free * maturity) * (futures_price * norm.cdf(d1) - strike_price * norm.cdf(d2))
    elif options_type in ['put', 'PUT', 'Put']:
        return np.exp(-risk_free * maturity) * (strike_price * norm.cdf(-d2) - futures_price * norm.cdf(-d1))


def caplet(principle, risk_free, forward_rate, strike_price, sigma, time1, time2):
    """
    calculcate interests cap price
    :param principle: principle on caplet
    :param risk_free: risk free rate
    :param forward_rate: forward rate
    :param strike_price: strike price
    :param sigma: volatility
    :param time1: time 1
    :param time2: time 2
    :return: price of caplet
    """
    d1 = (np.log(forward_rate / strike_price) + 0.2 * pow(sigma, 2) * time1) / (sigma * np.sqrt(time1))
    d2 = d1 - sigma * np.sqrt(time1)
    tau = time2 - time1
    return principle * tau * np.exp(-risk_free * time2) * (forward_rate * norm.cdf(d1) - strike_price * norm.cdf(d2))


def floorlet(principle, risk_free, forward_rate, strike_price, sigma, time1, time2):
    """
    calculcate interests floorlet price
    :param principle: principle on caplet
    :param risk_free: risk free rate
    :param forward_rate: forward rate
    :param strike_price: strike price
    :param sigma: volatility
    :param time1: time 1
    :param time2: time 2
    :return: price of caplet
    """
    d1 = (np.log(forward_rate / strike_price) + pow(sigma, 2) * time1 / 2) / (sigma * np.sqrt(time1))
    d2 = d1 - sigma * np.sqrt(time1)
    tau = time2 - time1
    return principle * tau * np.exp(-risk_free * time2) * (strike_price * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))


def swapoption(principle, forward_rate, fix_rate, frequency, sigma, options_maturity, swap_maturity, risk_free_list,
               direction):
    """
    calculate swapoptin value
    :param principle: principle on swap
    :param forward_rate: forward rate
    :param fix_rate: fix rate
    :param frequency: payoof frequency
    :param sigma: volalitility
    :param options_maturity: options maturity
    :param swap_maturity: swap maturity
    :param risk_free_list: risk free rate list
    :param direction: holding direction
    :return: swapoption price
    """
    d1 = (np.log(forward_rate / fix_rate) + pow(sigma, 2) * options_maturity / 2) / (sigma * np.sqrt(options_maturity))
    d2 = d1 = sigma * np.sqrt(options_maturity)
    t_list = options_maturity + np.arange(1, frequency * swap_maturity + 1) / frequency
    if direction == 'pay':
        value = np.sum(np.exp(-risk_free_list * t_list) * principle * (
                forward_rate * norm.cdf(d1) - fix_rate * norm.cdf(d2)) / frequency)
    else:
        value = np.sum(np.exp(-risk_free_list * t_list) * principle * (
                fix_rate * norm.cdf(-d2) - forward_rate * norm.cdf(-d1)) / frequency)
    return value


def foward_swaprates(swap_rate_list, options_maturity, swap_maturity, frequency):
    """
    calculate forward swap rates
    :param swap_rate_list: swap rate list
    :param options_maturity: options maturity
    :param frequency: payoff frequency
    :return: swap rates forward
    """
    t_list = frequency * np.arange(1, frequency * swap_maturity + 1)
    a = pow(1 + swap_rate_list[0] / frequency, -frequency * options_maturity) - pow(1 + swap_rate_list[-1] / frequency,
                                                                                    -frequency * (
                                                                                            options_maturity + swap_maturity))
    b=(1/options_maturity)*np.sum(pow(1+swap_rate_list[1:]/frequency,-t_list))
    return a/b


if __name__ == "__main__":
    print(pd_merton(21.85, 63.90, 82.6259, 0.1124, 0.050001, 1))
    print(convert_bond_binal_tree(50, 0.2, 100, 2, 0.01, 0.05, 0.4, 110, 9 / 12, 300))
    print(bsm_futures_options(420.36, 380, 0.1757, 0.02697, 0.20273972602739726, 'call'))
    print(bsm_futures_options(420.36, 380, 0.1757, 0.02697, 0.20273972602739726, 'put'))
    print(caplet(1e8, 0.017049, 0.0287, 0.022, 0.064905, 3 / 12, 6 / 12))
    print(floorlet(1e8, 0.017049, 0.0287, 0.025, 0.064905, 3 / 12, 6 / 12))
