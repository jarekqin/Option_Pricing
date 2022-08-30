import numpy as np
from scipy.stats import norm


def BSM_ORIGINAL(underlying_price, strike_price, maturity, rate, vol, options_type='call'):
    """
    BSM options original version
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :param options_type: 'call'/'put'
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (rate + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)

    if options_type.lower() == 'call':
        return underlying_price * norm.cdf(d1) - strike_price * np.exp(-rate * maturity) * norm.cdf(d2)
    elif options_type.lower() == 'put':
        return strike_price * np.exp(-rate * maturity) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
    else:
        raise NotImplemented


def BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
    """
    BSM options with carry_cost version
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :param options_type: 'call'/'put'
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)

    if options_type.lower() == 'call':
        return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(
            d1) - strike_price * np.exp(
            -rate * maturity) * norm.cdf(d2)
    elif options_type.lower() == 'put':
        return strike_price * np.exp(-rate * maturity) * norm.cdf(-d2) - underlying_price * np.exp(
            (carry_cost - rate) * maturity) * norm.cdf(-d1)
    else:
        raise NotImplemented
    
    


def BSM_DELTA(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
    """
    BSM options delta greek
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :param options_type: 'call'/'put'
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))
    if options_type.lower() == 'call':
        return np.exp((carry_cost - rate) * maturity) * norm.cdf(d1)
    elif options_type.lower() == 'put':
        return np.exp((carry_cost - rate) * maturity) * (norm.cdf(d1) - 1)
    else:
        raise NotImplemented


def BSM_GAMMA(underlying_price, strike_price, maturity, rate, carry_cost, vol):
    """
    BSM options gamma greek
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))

    return np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) / (
                underlying_price * vol * np.sqrt(maturity))


def BSM_VEGA(underlying_price, strike_price, maturity, rate, carry_cost, vol):
    """
    BSM options vega greek
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))

    return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) * np.sqrt(maturity)


def BSM_THETA(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
    """
    BSM options theta greek
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :param options_type: 'call'/'put'
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)

    if options_type.lower() == 'call':
        return -underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) * vol / (
                2 * np.sqrt(maturity)) - (carry_cost - rate) * underlying_price * np.exp(
            (carry_cost - rate) * maturity) * norm.cdf(d1) - rate * strike_price * np.exp(
            -rate * maturity) * norm.cdf(d2)
    elif options_type.lower() == 'put':
        return -underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) * vol / (
                2 * np.sqrt(maturity)) + (carry_cost - rate) * underlying_price * np.exp(
            (carry_cost - rate) * maturity) * norm.cdf(-d1) + rate * strike_price * np.exp(
            -rate * maturity) * norm.cdf(-d2)
    else:
        raise NotImplemented

def BSM_RHO(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
    """
    BSM options rho greek
    :param underlying_price: underlying price on optins
    :param strike_price: strike price of options
    :param maturity: options maturity
    :param rate: risk-free rate
    :param carry_cost: carry out fee
    :param vol: volatility
    :param options_type: 'call'/'put'
    :return: options price
    """
    d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
            vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)
    if options_type.lower() == 'call':
        if carry_cost != 0:
            return maturity * strike_price * np.exp(-rate * maturity) * norm.cdf(d2)
        else:
            return -maturity * BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, 'call')
    elif options_type.lower() == 'put':
        if carry_cost != 0:
            return -maturity * strike_price * np.exp(-rate * maturity) * norm.cdf(-d2)
        else:
            return -maturity * BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, 'put')
    else:
        raise NotImplemented


if __name__=='__main__':
    print(BSM(60, 65, 0.25, 0.08, 0.08, 0.3, 'call'))
    print(BSM(60, 65, 0.25, 0.08, 0.08, 0.3, 'put'))
    print('call delta', BSM_DELTA(60, 65, 0.25, 0.08, 0.08, 0.3, 'call'))
    print('put delta', BSM_DELTA(60, 65, 0.25, 0.08, 0.08, 0.3, 'put'))
    print('gamma', BSM_GAMMA(60, 65, 0.25, 0.08, 0.08, 0.3))
    print('vega', BSM_VEGA(60, 65, 0.25, 0.08, 0.08, 0.3))
    print('call theta', BSM_THETA(60, 65, 0.25, 0.08, 0.08, 0.3, 'call'))
    print('put theta', BSM_THETA(60, 65, 0.25, 0.08, 0.08, 0.3, 'put'))
    print('call rho', BSM_RHO(60, 65, 0.25, 0.08, 0.08, 0.3, 'call'))
    print('put rho', BSM_RHO(60, 65, 0.25, 0.08, 0.08, 0.3, 'put'))