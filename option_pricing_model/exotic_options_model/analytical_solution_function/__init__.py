import numpy as np

from scipy.stats import norm

from option_pricing_model.vanilla_options_model.analytic_model.bsm_based_class import BSMBase


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


if __name__ == '__main__':
    print('execution stock call options', ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15))
    print('execution stock put options',
          ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15, 'put'))
    print('forward call options', ExoticOPtions.Forward_Options(60, 1.1, 0.25, 1, 0.08, 0.04, 0.3))
    print('forward put options', ExoticOPtions.Forward_Options(60, 1.1, 0.25, 1, 0.08, 0.04, 0.3, 'put'))
    print('time switch call options', ExoticOPtions.Time_Switch_Options(100, 110, 5, 1, 0, 0.00274, 0.06, 0.06, 0.26))
    print('time switch put options',
          ExoticOPtions.Time_Switch_Options(100, 110, 5, 1, 0, 0.00274, 0.06, 0.06, 0.26, 'put'))
