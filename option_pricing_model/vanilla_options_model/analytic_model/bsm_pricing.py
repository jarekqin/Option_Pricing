from option_pricing_model.options_tools.tools import *
from option_pricing_model.vanilla_options_model.bsm_base import  BSM_ORIGINAL

import numpy as np

class BSMPricingModule(object):
    @staticmethod
    def BSM_CARRAY(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
        """
        BSM options based on carry cost
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
            return maturity * underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1)
        elif options_type.lower() == 'put':
            return -maturity * underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1)
        else:
            raise NotImplemented

    @staticmethod
    def BSM_FRENCH(underlying_price, strike_price, trading_time, calendar_time, rate, carry_cost, vol,
                   options_type='call'):
        """
        BSM options based on FRENCH formula
        :param underlying_price: underlying price on optins
        :param strike_price: strike price of options
        :param trading_time: options trading_time
        :param calendar_time: options calendar_time
        :param rate: risk-free rate
        :param carry_cost: carry out fee
        :param vol: volatility
        :param options_type: 'call'/'put'
        :return: options price
        """
        d1 = (np.log(underlying_price / strike_price) + carry_cost * trading_time + vol ** 2 / 2 * calendar_time) / (
                vol * np.sqrt(calendar_time))
        d2 = d1 - vol * np.sqrt(calendar_time)

        if options_type.lower() == 'call':
            return underlying_price * np.exp((carry_cost - rate) * trading_time) * norm.cdf(
                d1) - strike_price * np.exp(
                -rate * trading_time) * norm.cdf(d2)
        elif options_type.lower() == 'put':
            return strike_price * np.exp(-rate * trading_time) * norm.cdf(-d2) - underlying_price * np.exp(
                (carry_cost - rate) * trading_time) * norm.cdf(-d1)
        else:
            raise NotImplemented

    @staticmethod
    def BSM_JUMP_DIFFUS(underlying_price, strike_price, maturity, rate, vol, lambda1, gamma, options_type='call',
                        step=10):
        """
        BSM options pricing model with jumps
        :param underlying_price: underlying price on optins
        :param strike_price: strike price of options
        :param maturity: options maturity
        :param rate: risk-free rate
        :param dividend_payout_time: dividends payoff time
        :param vol: volatility
        :param lambda1: x axis value
        :param gamma: y axis value
        :param step: grid steps for calculation
        :return: options price
        """
        delta = np.sqrt(gamma * vol ** 2 / lambda1)
        z = np.sqrt(vol ** 2 - lambda1 * delta ** 2)
        result = 0
        for i in range(step):
            vi = np.sqrt(z ** 2 + delta ** 2 * (i / maturity))
            result += np.exp(-lambda1 * maturity) * (lambda1 * maturity) ** i / np.math.factorial(
                i) * BSM(
                underlying_price, strike_price, maturity, rate, rate, vi, options_type)
        return result

    @staticmethod
    def BSM_USA(underlying_price, strike_price, dividend_payout_time, maturity, rate, dividend, vol):
        """
        American options BSM based on dividends payment
        :param underlying_price: underlying price on optins
        :param strike_price: strike price of options
        :param maturity: options maturity
        :param rate: risk-free rate
        :param dividend_payout_time: dividends payoff time
        :param vol: volatility
        :return: options price
        """
        infinity = 1e8
        epsilon = 1e-5
        sx = underlying_price - dividend * np.exp(-rate * dividend_payout_time)
        if dividend <= strike_price * (1 - np.exp(-rate * (maturity - dividend_payout_time))):
            return BSM_ORIGINAL(sx, strike_price, maturity, rate, vol, 'call')
        ci = BSM_ORIGINAL(underlying_price, strike_price, maturity - dividend_payout_time, rate, vol, 'call')
        highs = underlying_price
        while (ci - highs - dividend + strike_price) > 0 and highs < infinity:
            highs *= 2
            ci = BSM_ORIGINAL(highs, strike_price, maturity - dividend_payout_time, rate, vol, 'call')
        if highs > infinity:
            return BSM_ORIGINAL(sx, strike_price, maturity, rate, vol, 'call')

        lows = 0
        i = highs * 0.5
        ci = BSM_ORIGINAL(i, strike_price, maturity - dividend_payout_time, rate, vol, 'call')
        while abs(ci - i - dividend + strike_price) > epsilon and highs - lows > epsilon:
            if (ci - i - dividend + strike_price) < 0:
                highs = i
            else:
                lows = i
            i = (highs + lows) / 2
            ci = BSM_ORIGINAL(i, strike_price, maturity - dividend_payout_time, rate, vol, 'call')
        a1 = (np.log(sx / strike_price) + (rate + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
        a2 = a1 - vol * np.sqrt(maturity)
        b1 = (np.log(sx / i) + (rate + vol ** 2 / 2) * dividend_payout_time) / (vol * np.sqrt(dividend_payout_time))
        b2 = b1 - vol * np.sqrt(dividend_payout_time)

        return sx * norm.cdf(b1) + sx * CBND(a1, -b1, -np.sqrt(dividend_payout_time / maturity)) - \
               strike_price * np.exp(-rate * maturity) * CBND(a2, -b2,
                                                                    -np.sqrt(dividend_payout_time / maturity)) - \
               (strike_price - dividend) * np.exp(-rate * dividend_payout_time) * norm.cdf(b2)

    @staticmethod
    def BSM_USA_BAAPPROX(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
        """
        American options approximation method based on BS formula
        :param underlying_price: underlying price on optins
        :param strike_price: strike price of options
        :param maturity: options maturity
        :param rate: risk-free rate
        :param carry_cost: carry out fee
        :param vol: volatility
        :param options_type: 'call'/'put'
        :return: options price
        """
        if options_type.lower() == 'call':
            if carry_cost >= rate:
                return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type)
            else:
                sk = Kc(strike_price, maturity, rate, carry_cost, vol)
                n = 2 * carry_cost / vol ** 2
                k = 2 * rate / (vol ** 2 * (1 - np.exp(-rate * maturity)))
                d1 = (np.log(sk / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                q2 = (-(n - 1) + np.sqrt((n - 1) ** 2 + 4 * k)) / 2
                a2 = (sk / q2) * (1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(d1))
                if underlying_price < sk:
                    return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol) + a2 * (
                            underlying_price / sk) ** q2
                else:
                    return underlying_price - strike_price

        elif options_type.lower() == 'put':
            sk = kp(strike_price, maturity, rate, carry_cost, vol)
            n = 2 * carry_cost / vol ** 2
            k = 2 * rate / (vol ** 2 * (1 - np.exp(-rate * maturity)))
            d1 = (np.log(sk / strike_price) + (carry_cost + vol ** 2) * maturity) / (vol * np.sqrt(maturity))
            q1 = (-(n - 1) - np.sqrt((n - 1) ** 2 + 4 * k)) / 2
            a1 = -(sk / q1) * (1 - np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1))

            if underlying_price > sk:
                return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol,
                                   options_type) + a1 * (underlying_price / sk) ** q1
            else:
                return strike_price - underlying_price

    @staticmethod
    def BSM_USA_BSAPPROX(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type='call'):
        """
        American options approximation method based on BS formula
        :param underlying_price: underlying price on optins
        :param strike_price: strike price of options
        :param maturity: options maturity
        :param rate: risk-free rate
        :param carry_cost: carry out fee
        :param vol: volatility
        :param options_type: 'call'/'put'
        :return: options price
        """
        if options_type.lower() == 'call':
            if carry_cost >= rate:
                return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type)
            else:
                beta = (1 / 2 - carry_cost / vol ** 2) + np.sqrt(
                    (carry_cost / vol ** 2 - 1 / 2) ** 2 + 2 * rate / vol ** 2)
                binfinity = beta / (beta - 1) * strike_price
                b0 = max(strike_price, rate / (rate - carry_cost) * strike_price)
                ht = -(carry_cost * maturity + 2 * vol * np.sqrt(maturity)) * b0 / (binfinity - b0)
                I = b0 + (binfinity - b0) * (1 - np.exp(ht))
                alpha = (I - strike_price) * I ** (-beta)
                if underlying_price > I:
                    return underlying_price - strike_price
                else:
                    return alpha * underlying_price ** beta - alpha * phi(underlying_price, maturity, beta, I, I, rate,
                                                                          carry_cost,
                                                                          vol) + \
                           phi(underlying_price, maturity, 1, I, I, rate, carry_cost, vol) - phi(
                        underlying_price, maturity, 1, strike_price, I, rate, carry_cost, vol) - \
                           strike_price * phi(underlying_price, maturity, 0, I, I, rate, carry_cost, vol) + \
                           strike_price * phi(underlying_price, maturity, 0, strike_price, I, rate, carry_cost, vol)
        elif options_type.lower() == 'put':
            rate = rate - carry_cost
            carry_cost = -carry_cost
            beta = (1 / 2 - carry_cost / vol ** 2) + np.sqrt(
                (carry_cost / vol ** 2 - 1 / 2) ** 2 + 2 * rate / vol ** 2)
            binfinity = beta / (beta - 1) * underlying_price
            b0 = max(underlying_price, rate / (rate - carry_cost) * underlying_price)
            ht = -(carry_cost * maturity + 2 * vol * np.sqrt(maturity)) * b0 / (binfinity - b0)
            I = b0 + (binfinity - b0) * (1 - np.exp(ht))
            alpha = (I - underlying_price) * I ** (-beta)
            if strike_price > I:
                return strike_price - underlying_price
            else:
                return alpha * strike_price ** beta - alpha * phi(strike_price, maturity, beta, I, I, rate, carry_cost,
                                                                  vol) + \
                       phi(strike_price, maturity, 1, I, I, rate, carry_cost, vol) - phi(
                    strike_price, maturity, 1, underlying_price, I, rate, carry_cost, vol) - \
                       underlying_price * phi(strike_price, maturity, 0, I, I, rate, carry_cost, vol) + \
                       underlying_price * phi(strike_price, maturity, 0, underlying_price, I, rate, carry_cost, vol)

    @staticmethod
    def BSM_MSCOMMODITY_OPTION(pt, ft, strike_price, t1, t2, vs, ve, vf, rhose, rhosf,
                               rhoef, kappae, kappaf, options_type='call'):
        """
        BSM based commodity options pricing model
        :param pt: zero_bond_price
        :param ft: futures_price
        :param strike_price: strike_price
        :param t1: option_maturity
        :param t2: futures contract_maturity
        :param vs: spot_commodity_vol
        :param ve: convenience_yield_vol
        :param vf: forward_rate
        :param rhose: rho_commodity_convenience_yield
        :param rhosf: rho_commodity_forwad_rate
        :param rhoef: rho_convenience_yield_forward_rate
        :param kappae: mean_reversion_convenience_yield
        :param kappaf: mean_reversion_forward_rate
        :return: commidity options price
        """

        vz = vs ** 2 * t1 + 2 * vs * (vf * rhosf * 1 / kappaf * (
                t1 - 1 / kappaf * np.exp(-kappaf * t2) * (np.exp(kappaf * t1) - 1)) - ve * rhose * 1 / kappae * (
                                              t1 - 1 / kappae * np.exp(-kappae * t2) * (np.exp(kappae * t1) - 1)))
        + ve ** 2 * 1 / kappae ** 2 * (t1 + 1 / (2 * kappae) * np.exp(-2 * kappae * t2) * (
                np.exp(2 * kappae * t1) - 1) - 2 * 1 / kappae * np.exp(-kappae * t2) * (
                                                   np.exp(kappae * t1) - 1)) \
        + vf ** 2 * 1 / kappaf ** 2 * (
                t1 + 1 / (2 * kappaf) * np.exp(-2 * kappaf * t2) * (
                np.exp(2 * kappaf * t1) - 1) - 2 * 1 / kappaf * np.exp(
            -kappaf * t2) * (np.exp(kappaf * t1) - 1)) - 2 * ve * vf * rhoef * 1 / kappae * 1 / kappaf * (
                t1 - 1 / kappae * np.exp(-kappae * t2) * (np.exp(kappae * t1) - 1) - 1 / kappaf * np.exp(
            -kappaf * t2) * (
                        np.exp(kappaf * t1) - 1) + 1 / (kappae + kappaf) * np.exp(-(kappae + kappaf) * t2) * (
                        np.exp((kappae + kappaf) * t1) - 1))

        vxz = vf * 1 / kappaf * (vs * rhosf * (t1 - 1 / kappaf * (1 - np.exp(-kappaf * t1))) + vf * 1 / kappaf * (
                    t1 - 1 / kappaf * np.exp(-kappaf * t2) * (np.exp(kappaf * t1) - 1) - 1 / kappaf * (
                    1 - np.exp(-kappaf * t1)) + 1 / (2 * kappaf) * np.exp(-kappaf * t2) * (
                                np.exp(kappaf * t1) - np.exp(-kappaf * t1))) - ve * rhoef * 1 / kappae * (
                                             t1 - 1 / kappae * np.exp(-kappae * t2) * (
                                             np.exp(kappae * t1) - 1) - 1 / kappaf * (
                                                     1 - np.exp(-kappaf * t1)) + 1 / (kappae + kappaf) * np.exp(
                                         -kappae * t2) * (np.exp(kappae * t1) - np.exp(-kappaf * t1))))

        vz = np.sqrt(vz)
        d1 = (np.log(ft / strike_price) - vxz + vz ** 2 / 2) / vz
        d2 = (np.log(ft / strike_price) - vxz - vz ** 2 / 2) / vz

        if options_type.lower() == 'call':
            return pt * (ft * np.exp(-vxz) * norm.cdf(d1) - strike_price * norm.cdf(d2))
        elif options_type.lower() == 'put':
            return pt * (strike_price * norm.cdf(-d2) - ft * np.exp(-vxz) * norm.cdf(-d1))
        else:
            raise NotImplemented


if __name__ == '__main__':
    print('call carry', BSMPricingModule.BSM_CARRAY(60, 65, 0.25, 0.08, 0.08, 0.3, 'call'))
    print('put carry', BSMPricingModule.BSM_CARRAY(60, 65, 0.25, 0.08, 0.08, 0.3, 'put'))
    print('call french', BSMPricingModule.BSM_FRENCH(70, 75, 0.41, 0.4, 0.08, 0.08, 0.3, 'call'))
    print('put french', BSMPricingModule.BSM_FRENCH(70, 75, 0.41, 0.4, 0.08, 0.08, 0.3, 'put'))
    print('call jump', BSMPricingModule.BSM_JUMP_DIFFUS(45, 55, 0.25, 0.1, 0.25, 4, 0.4, 'call'))
    print('put jump', BSMPricingModule.BSM_JUMP_DIFFUS(45, 55, 0.25, 0.1, 0.25, 4, 0.4, 'put'))
    print('usa option', BSMPricingModule.BSM_USA(90, 80, 0.25, 0.33, 0.06, 4, 0.3))
    print('ba usa call appro', BSMPricingModule.BSM_USA_BAAPPROX(42, 40, 0.75, 0.04, -0.04, 0.35))
    print('ba usa put appro', BSMPricingModule.BSM_USA_BAAPPROX(42, 40, 0.75, 0.04, -0.04, 0.35, 'put'))
    print('bs usa call appro', BSMPricingModule.BSM_USA_BSAPPROX(42, 40, 0.75, 0.04, -0.04, 0.35))
    print('bs usa put appro', BSMPricingModule.BSM_USA_BSAPPROX(42, 40, 0.75, 0.04, -0.04, 0.35, 'put'))
    print('commodity call',
          BSMPricingModule.BSM_MSCOMMODITY_OPTION(0.9753, 95, 95, 0.5, 1, 0.266, 0.249, 0.0096, 0.8050, 0.0964, 0.1243,
                                         1.045, 0.2))
    print('commodity put',
          BSMPricingModule.BSM_MSCOMMODITY_OPTION(0.9753, 95, 95, 0.5, 1, 0.266, 0.249, 0.0096, 0.8050, 0.0964, 0.1243,
                                         1.045, 0.2, 'put'))
