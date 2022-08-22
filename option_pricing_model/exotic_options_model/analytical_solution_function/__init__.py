import numpy as np

from scipy.stats import norm

from option_pricing_model.options_fools.tools import *


class ExoticOPtions(object):
    @staticmethod
    def Executive_Stock_Options(underlying_price, strike_price, maturity, rate, carry_cost, vol, lambda1,
                                options_type='call'):
        """
        Executive stock options
        :param underlying_price: underlying_price
        :param strike_price: strike_price
        :param maturity: options maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param lambda1: lambda factor
        :param options_type: call/put
        :return: options price
        """
        return np.exp(- lambda1 * maturity) * BSMBase.BSM(underlying_price, strike_price, maturity, rate, carry_cost,
                                                          vol, options_type)

    @staticmethod
    def Forward_Options(underlying_price, alpha, forward_time, maturity, rate, carry_cost, vol, options_type='call'):
        """
        Forward options
        :param underlying_price: underlying_price
        :param alpha: options price deplay factor
        :param forward_time: options can be written future time
        :param maturity: options maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        return underlying_price * np.exp((carry_cost - rate) * forward_time) * BSMBase.BSM(
            1, alpha, maturity - forward_time, rate, carry_cost, vol, options_type)

    @staticmethod
    def Time_Switch_Options(underlying_price, strike_price, accumulated_amount, maturity, time_units, time_interval,
                            rate, carry_cost, vol, options_type='call'):
        """
        Time switch options: give buyder side serveral time units that can be decided to call the options or not
        :param underlying_price: underlying_price
        :param strike_price: strike_price
        :param accumulated_amount: accumulated amount of underlyings
        :param maturity: options maturity
        :param time_units: given time units
        :param time_interval: interval among every 2 time units
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volalitility
        :param options_type: call/put
        :return: options price
        """
        time_step = int(np.ceil(maturity / time_interval))
        result = 0
        z_symbol = 1 if options_type.lower() == 'call' else -1

        for i in range(1, time_step + 1):
            d = (np.log(underlying_price / strike_price) + (carry_cost - vol ** 2 / 2) * i * time_interval) / (
                    vol * np.sqrt(i * time_interval))
            result += norm.cdf(z_symbol * d) * time_interval
        return accumulated_amount * np.exp(-rate * maturity) * result + time_interval * accumulated_amount * np.exp(
            -rate * maturity) * time_units

    @staticmethod
    def Chooser_Options(underlying_price, strike_price_call, strike_price_put, chooser_time, maturity_call,
                        maturity_put, rate, carry_cost, vol, chooser_type='simple'):
        """
        Chooser options for both 'simple' and 'complex
        :param underlying_price: underlying price
        :param strike_price_call: available both for 2 types. If type is 'simple' only use this strike price
        :param strike_price_put:  only available for 'complex' chooser options
        :param chooser_time: chooser options maturity
        :param maturity_call: common call options maturity
        :param maturity_put: common put options maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param chooser_type:simple/complex
        :return: chooser options price
        """
        if chooser_type.lower() == 'simple':
            d = (np.log(underlying_price / strike_price_call) + (carry_cost + vol ** 2 / 2) * maturity_call) / (
                    vol * np.sqrt(maturity_call))
            y = (np.log(
                underlying_price / strike_price_call) + carry_cost * maturity_call + vol ** 2 * chooser_time / 2) / (
                        vol * np.sqrt(chooser_time))

            return underlying_price * np.exp((carry_cost - rate) * maturity_call) * norm.cdf(
                d) - strike_price_call * np.exp(
                -rate * maturity_call) * norm.cdf(d - vol * np.sqrt(maturity_call)) - underlying_price * np.exp(
                (carry_cost - rate) * maturity_call) * norm.cdf(-y) + strike_price_call * np.exp(
                -rate * maturity_call) * norm.cdf(
                -y + vol * np.sqrt(chooser_time))
        elif chooser_type.lower() == 'complex':
            I = CriticalValueChooser(underlying_price, strike_price_call, strike_price_put, chooser_time, maturity_call,
                                     maturity_put, rate, carry_cost, vol)
            d1 = (np.log(underlying_price / I) + (carry_cost + vol ** 2 / 2) * chooser_time) / (
                    vol * np.sqrt(chooser_time))
            d2 = d1 - vol * np.sqrt(chooser_time)
            y1 = (np.log(underlying_price / strike_price_call) + (carry_cost + vol ** 2 / 2) * maturity_call) / (
                    vol * np.sqrt(maturity_call))
            y2 = (np.log(underlying_price / strike_price_put) + (carry_cost + vol ** 2 / 2) * maturity_put) / (
                    vol * np.sqrt(maturity_put))
            rho1 = np.sqrt(chooser_time / maturity_call)
            rho2 = np.sqrt(chooser_time / maturity_put)

            return underlying_price * np.exp((carry_cost - rate) * maturity_call) * CBND(d1, y1, rho1) - \
                   strike_price_call * np.exp(-rate * maturity_call) * CBND(d2, y1 - vol * np.sqrt(maturity_call), rho1) \
                   - underlying_price * np.exp((carry_cost - rate) * maturity_put) * CBND(-d1, -y2, rho2) + \
                   strike_price_put * np.exp(-rate * maturity_put) * CBND(-d2, -y2 + vol * np.sqrt(maturity_put), rho2)

        else:
            raise NotImplemented


if __name__ == '__main__':
    print('execution stock call options', ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15))
    print('execution stock put options',
          ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15, 'put'))
    print('forward call options', ExoticOPtions.Forward_Options(60, 1.1, 0.25, 1, 0.08, 0.04, 0.3))
    print('forward put options', ExoticOPtions.Forward_Options(60, 1.1, 0.25, 1, 0.08, 0.04, 0.3, 'put'))
    print('time switch call options', ExoticOPtions.Time_Switch_Options(100, 110, 5, 1, 0, 0.00274, 0.06, 0.06, 0.26))
    print('time switch put options',
          ExoticOPtions.Time_Switch_Options(100, 110, 5, 1, 0, 0.00274, 0.06, 0.06, 0.26, 'put'))
    print('simple chooser options', ExoticOPtions.Chooser_Options(50, 55, 0, 0.25, 0.5, 0, 0.08, 0.08, 0.25))
    print('complex chooser options',
          ExoticOPtions.Chooser_Options(50, 55, 48, 0.25, 0.5, 0.5833, 0.1, 0.05, 0.35, 'complex'))
