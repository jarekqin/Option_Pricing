import numpy as np

from scipy.stats import norm

from option_pricing_model.options_fools.tools import CriticalValueChooser,CriticalValueOptionsOnOptions
from option_pricing_model.vanilla_options_model.analytic_model.bsm_models import BSMBase,CBND


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

    @staticmethod
    def Options_On_Options(underlying_price, strike_price1, strike_price2, maturity1, maturity2, rate, carry_cost, vol,
                           options_type='cc'):
        """
        options on options pricing model
        :param underlying_price: underlying price
        :param strike_price1: options 1 strike price
        :param strike_price2: convered options strike price
        :param maturity1: options 1 maturity
        :param maturity2: convered options maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        if options_type.lower() == 'cc' or options_type.lower() == 'pc':
            underlying_options_type = 'call'
        else:
            underlying_options_type = 'put'

        I = CriticalValueOptionsOnOptions(strike_price1, strike_price2, maturity2 - maturity1, rate,
                                          carry_cost, vol, underlying_options_type)

        rho = np.sqrt(maturity1 / maturity2)
        y1 = (np.log(underlying_price / I) + (carry_cost + vol ** 2 / 2) * maturity1) / (vol * np.sqrt(maturity1))
        y2 = y1 - vol * np.sqrt(maturity1)
        z1 = (np.log(underlying_price / strike_price1) + (carry_cost + vol ** 2 / 2) * maturity2) / (
                vol * np.sqrt(maturity2))
        z2 = z1 - vol * np.sqrt(maturity2)

        if options_type.lower() == 'cc':
            return underlying_price * np.exp((carry_cost - rate) * maturity2) * CBND(z1, y1, rho) - strike_price1 * \
                   np.exp(-rate * maturity2) * CBND(z2, y2, rho) - strike_price2 * np.exp(-rate * maturity1) * \
                   norm.cdf(y2)
        elif options_type.lower() == 'pc':
            return strike_price1 * np.exp(-rate * maturity2) * CBND(z2, -y2, -rho) - underlying_price * \
                   np.exp((carry_cost - rate) * maturity2) * CBND(z1, -y1, -rho) + strike_price2 * \
                   np.exp(-rate * maturity1) * norm.cdf(-y2)
        elif options_type.lower() == 'cp':
            return strike_price1 * np.exp(-rate * maturity2) * CBND(-z2, -y2, rho) - underlying_price * \
                   np.exp((carry_cost - rate) * maturity2) * CBND(-z1, -y1, rho) - strike_price2 * \
                   np.exp(-rate * maturity1) * norm.cdf(-y2)
        elif options_type.lower() == 'pp':
            return underlying_price * np.exp((carry_cost - rate) * maturity2) * CBND(-z1, y1, -rho) - strike_price1 * \
                   np.exp(-rate * maturity2) * CBND(-z2, y2, -rho) + np.exp(-rate * maturity1) * strike_price2 * \
                   norm.cdf(y2)
        else:
            raise TypeError

    @staticmethod
    def Extendible_Options(
            underlying_price, strike_price, extendible_strike_price, maturity, extendible_maturity,
            rate, carry_cost, vol, options_type):
        """
        extendible options pricing module
        :param underlying_price: underlying price
        :param strike_price: strike price on options
        :param extendible_strike_price: extendible strike price
        :param maturity: maturity on optins
        :param extendible_maturity: extendible maturity on options
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        rho = np.sqrt(maturity / extendible_maturity)
        z1 = (np.log(underlying_price / extendible_strike_price) + (
                carry_cost + vol ** 2 / 2) * extendible_maturity) / (vol * np.sqrt(extendible_maturity))
        z2 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        if options_type.lower() == 'call':
            return BSMBase.BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type) + \
                   underlying_price * np.exp((carry_cost - rate) * extendible_maturity) * CBND(z1, -z2, -rho) - \
                   extendible_strike_price * np.exp(-rate * extendible_maturity) * CBND(
                z1 - np.sqrt(vol ** 2 * extendible_maturity),
                -z2 + np.sqrt(vol ** 2 * maturity), -rho)
        elif options_type.lower() == 'put':
            return BSMBase.BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type) + \
                   extendible_strike_price * np.exp(-rate * extendible_maturity) * CBND(
                -z1 + np.sqrt(vol ** 2 * extendible_maturity),
                z2 - np.sqrt(vol ** 2 * maturity), -rho) - \
                   underlying_price * np.exp((carry_cost - rate) * extendible_maturity) * CBND(-z1, z2, -rho)
        else:
            raise NotImplemented

    @staticmethod
    def Two_Asset_Corr_Options(underlying_price1, underlying_price2, strike_price1, strike_price2, maturity,
                               rate, carry_cost1, carry_cost2, vol1, vol2, correlation, options_type):
        """
        2 asset correlation options
        :param underlying_price1: underlying asset 1 price
        :param underlying_price2: underlying asset 2 price
        :param strike_price1: strike price 1
        :param strike_price2: strike price 2
        :param maturity: maturity
        :param rate: risk-free rate
        :param carry_cost1: carry cost 1
        :param carry_cost2: carry cost 2
        :param vol1: volatility 1
        :param vol2: volatitlity 2
        :param correlation: correlation between 2 assets
        :param options_type: call/put
        :return: options price
        """
        y1 = (np.log(underlying_price1 / strike_price1) + (carry_cost1 - vol1 ** 2 / 2) * maturity) / (
                vol1 * np.sqrt(maturity))
        y2 = (np.log(underlying_price2 / strike_price2) + (carry_cost2 - vol2 ** 2 / 2) * maturity) / (
                vol2 * np.sqrt(maturity))
        if options_type.lower() == 'call':
            return underlying_price2 * np.exp((carry_cost2 - rate) * maturity) * CBND(y2 + vol2 * np.sqrt(maturity),
                                                                                      y1 + correlation * vol2 * np.sqrt(
                                                                                          maturity),
                                                                                      correlation) - strike_price2 * \
                   np.exp(-rate * maturity) * CBND(y2, y1, correlation)

        elif options_type.lower() == 'put':
            return strike_price2 * np.exp(-rate * maturity) * CBND(-y2, -y1, correlation) - \
                   underlying_price2 * np.exp((carry_cost2 - rate) * maturity) * CBND(-y2 - vol2 * np.sqrt(maturity),
                                                                                      -y1 - correlation * vol2 * np.sqrt(
                                                                                          maturity), correlation)
        else:
            raise NotImplemented

    @staticmethod
    def Asset_Exchange_Options(underlying_price1, underlying_price2, num_1, num_2, maturity, rate, carry_cost1,
                               carry_cost2, vol1, vol2, correlation, options_model):
        """
        asset exchange options
        :param underlying_price1:underlying asset 1 price
        :param underlying_price2: underlying asset 2 price
        :param num_1: quantity on asset 1
        :param num_2: quantity on asset 2
        :param maturity:maturity on options
        :param rate:risk-free rate
        :param carry_cost1:carry cost 1
        :param carry_cost2:carry cost 2
        :param vol1:volatility 1
        :param vol2:volatillity 2
        :param correlation:correlation between 2 assets
        :param options_model:eu/usa
        :return:
        """
        if options_model.lower() == 'eu':
            v = np.sqrt(vol1 ** 2 + vol2 ** 2 - 2 * correlation * vol1 * vol2)
            d1 = (np.log(num_1 * underlying_price1 / (num_2 * underlying_price2)) + (
                    carry_cost1 - carry_cost2 + v ** 2 / 2) * maturity) / (v * np.sqrt(maturity))
            d2 = d1 - v * np.sqrt(maturity)
            return num_1 * underlying_price1 * np.exp((carry_cost1 - rate) * maturity) * norm.cdf(
                d1) - num_2 * underlying_price2 * np.exp((carry_cost2 - rate) * maturity) * norm.cdf(d2)
        elif options_model.lower()=='usa':
            v = np.sqrt(vol1 ** 2 + vol2 ** 2 - 2 * correlation * vol1 * vol2)
            return BSMBase.BSM_USA_BSAPPROX(num_1*underlying_price1,num_2*underlying_price2,maturity,rate-carry_cost1,carry_cost1-carry_cost2,v,'call')
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
    print('cp options on options',
          ExoticOPtions.Options_On_Options(500, 520, 50, 0.25, 0.5, 0.08, 0.05, 0.35, 'cc'))
    print('cp options on options',
          ExoticOPtions.Options_On_Options(500, 520, 50, 0.25, 0.5, 0.08, 0.05, 0.35, 'cp'))
    print('pp options on options',
          ExoticOPtions.Options_On_Options(500, 520, 50, 0.25, 0.5, 0.08, 0.05, 0.35, 'pp'))
    print('pc options on options',
          ExoticOPtions.Options_On_Options(500, 520, 50, 0.25, 0.5, 0.08, 0.05, 0.35, 'pc'))
    print('extendible call options',
          ExoticOPtions.Extendible_Options(80, 90, 82, 0.5, 0.75, 0.1, 0.1, 0.3, 'call'))
    print('extendible put options',
          ExoticOPtions.Extendible_Options(80, 90, 82, 0.5, 0.75, 0.1, 0.1, 0.3, 'put'))
    print('2 assets correlation call options',
          ExoticOPtions.Two_Asset_Corr_Options(52, 65, 50, 70, 0.5, 0.1, 0.1, 0.1, 0.2, 0.3, 0.75, 'call'))
    print('extendible put options',
          ExoticOPtions.Two_Asset_Corr_Options(52, 65, 50, 70, 0.5, 0.1, 0.1, 0.1, 0.2, 0.3, 0.75, 'put'))
    print('asset exchange eu options',
          ExoticOPtions.Asset_Exchange_Options(101,104,1,1,0.5,0.1,0.02,0.04,0.18,0.12,0.8,'eu'))
    print('extendible put options',
          ExoticOPtions.Asset_Exchange_Options(101,104,1,1,0.5,0.1,0.02,0.04,0.18,0.12,0.8,'usa'))