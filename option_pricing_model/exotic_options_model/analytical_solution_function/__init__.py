import numpy as np

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
        :param rate:  risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param lambda1: lambda factor
        :param options_type: call/put
        :return: options price
        """
        return np.exp(- lambda1 * maturity) * BSMBase.BSM(underlying_price, strike_price, maturity, rate, carry_cost,
                                                          vol, options_type)


if __name__ == '__main__':
    print(ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15))
    print(ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15, 'put'))
