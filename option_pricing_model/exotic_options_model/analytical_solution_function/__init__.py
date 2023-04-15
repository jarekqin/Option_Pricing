from option_pricing_model.vanilla_options_model.bsm_base import BSM
from option_pricing_model.options_tools.tools import *


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
        :param lambda1: a_b_actual_extremum factor
        :param options_type: call/put
        :return: options price
        """
        return np.exp(- lambda1 * maturity) * BSM(underlying_price, strike_price, maturity, rate, carry_cost,
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
        return underlying_price * np.exp((carry_cost - rate) * forward_time) * BSM(
            1, alpha, maturity - forward_time, rate, carry_cost, vol, options_type)

    @staticmethod
    def maturityime_Switch_Options(underlying_price, strike_price, accumulated_amount, maturity, time_units,
                                   time_interval,
                                   rate, carry_cost, vol, options_type='call'):
        """
        maturityime switch options: give buyder side serveral time units that can be decided to call the options or not
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
            correlation1 = np.sqrt(chooser_time / maturity_call)
            correlation2 = np.sqrt(chooser_time / maturity_put)

            return underlying_price * np.exp((carry_cost - rate) * maturity_call) * CBND(d1, y1, correlation1) - \
                strike_price_call * np.exp(-rate * maturity_call) * CBND(d2, y1 - vol * np.sqrt(maturity_call),
                                                                         correlation1) \
                - underlying_price * np.exp((carry_cost - rate) * maturity_put) * CBND(-d1, -y2, correlation2) + \
                strike_price_put * np.exp(-rate * maturity_put) * CBND(-d2, -y2 + vol * np.sqrt(maturity_put),
                                                                       correlation2)

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

        correlation = np.sqrt(maturity1 / maturity2)
        y1 = (np.log(underlying_price / I) + (carry_cost + vol ** 2 / 2) * maturity1) / (vol * np.sqrt(maturity1))
        y2 = y1 - vol * np.sqrt(maturity1)
        z1 = (np.log(underlying_price / strike_price1) + (carry_cost + vol ** 2 / 2) * maturity2) / (
                vol * np.sqrt(maturity2))
        z2 = z1 - vol * np.sqrt(maturity2)

        if options_type.lower() == 'cc':
            return underlying_price * np.exp((carry_cost - rate) * maturity2) * CBND(z1, y1,
                                                                                     correlation) - strike_price1 * \
                np.exp(-rate * maturity2) * CBND(z2, y2, correlation) - strike_price2 * np.exp(-rate * maturity1) * \
                norm.cdf(y2)
        elif options_type.lower() == 'pc':
            return strike_price1 * np.exp(-rate * maturity2) * CBND(z2, -y2, -correlation) - underlying_price * \
                np.exp((carry_cost - rate) * maturity2) * CBND(z1, -y1, -correlation) + strike_price2 * \
                np.exp(-rate * maturity1) * norm.cdf(-y2)
        elif options_type.lower() == 'cp':
            return strike_price1 * np.exp(-rate * maturity2) * CBND(-z2, -y2, correlation) - underlying_price * \
                np.exp((carry_cost - rate) * maturity2) * CBND(-z1, -y1, correlation) - strike_price2 * \
                np.exp(-rate * maturity1) * norm.cdf(-y2)
        elif options_type.lower() == 'pp':
            return underlying_price * np.exp((carry_cost - rate) * maturity2) * CBND(-z1, y1,
                                                                                     -correlation) - strike_price1 * \
                np.exp(-rate * maturity2) * CBND(-z2, y2, -correlation) + np.exp(-rate * maturity1) * strike_price2 * \
                norm.cdf(y2)
        else:
            raise TypeError

    @staticmethod
    def Extendible_Options(
            underlying_price, strike_price, extendible_strike_price, maturity, extendible_maturity,
            rate, carry_cost, vol, options_type):
        correlation = np.sqrt(maturity / extendible_maturity)
        z1 = (np.log(underlying_price / extendible_strike_price) + (
                carry_cost + vol ** 2 / 2) * extendible_maturity) / (vol * np.sqrt(extendible_maturity))
        z2 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        if options_type.lower() == 'call':
            return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type) + \
                underlying_price * np.exp((carry_cost - rate) * extendible_maturity) * CBND(z1, -z2, -correlation) - \
                extendible_strike_price * np.exp(-rate * extendible_maturity) * CBND(
                    z1 - np.sqrt(vol ** 2 * extendible_maturity),
                    -z2 + np.sqrt(vol ** 2 * maturity), -correlation)
        elif options_type.lower() == 'put':
            return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type) + \
                extendible_strike_price * np.exp(-rate * extendible_maturity) * CBND(
                    -z1 + np.sqrt(vol ** 2 * extendible_maturity),
                    z2 - np.sqrt(vol ** 2 * maturity), -correlation) - \
                underlying_price * np.exp((carry_cost - rate) * extendible_maturity) * CBND(-z1, z2, -correlation)
        else:
            raise NotImplemented

    @staticmethod
    def maturitywo_Asset_Corr_Options(underlying_price1, underlying_price2, strike_price1, strike_price2, maturity,
                                      rate, carry_cosmaturity1, carry_cost2, vol1, vol2, correlation, options_type):
        y1 = (np.log(underlying_price1 / strike_price1) + (carry_cosmaturity1 - vol1 ** 2 / 2) * maturity) / (
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
    def Exchange_On_Exchange_Options(underlying_price1, underlying_price2, quantity_on_asset2, maturity1, maturity2,
                                     rate, carry_cost1, carry_cost2, vol1, vol2, correlation, options_type):
        """
        exchange options on exchange options
        :param underlying_price1: asset 1 price
        :param underlying_price2: asset 2 price
        :param quantity_on_asset2: quantity on asset 2
        :param maturity1: maturity on exchange options
        :param maturity2: maturity on exchange options of exchange options
        :param rate: risk-free risk
        :param carry_cost1: carry cost
        :param carry_cost2: carry cost
        :param vol1: volatitlitty 1
        :param vol2: volatitlity 2
        :param correlation: correlation between 2 assets
        :param options_type: int type 1-4
        1	[1] Option to exchange Q*underlying_price2 for the option to exchange underlying_price2 for underlying_price1
        2	[2] Option to exchange the option to exchange underlying_price2 for underlying_price1 in return for Q*underlying_price2
        3	[3] Option to exchange Q*underlying_price2 for the option to exchange underlying_price1 for underlying_price2
        4	[4] Option to exchange the option to exchange underlying_price1 for underlying_price2 in return for Q*underlying_price2
        :return: options price
        """
        v = np.sqrt(vol1 ** 2 + vol2 ** 2 - 2 * correlation * vol1 * vol2)
        I1 = underlying_price1 * np.exp((carry_cost1 - rate) * (maturity2 - maturity1)) / (
                underlying_price2 * np.exp((carry_cost2 - rate) * (maturity2 - maturity1)))
        if int(options_type) == 1 or int(options_type) == 2:
            id_ = 1
        else:
            id_ = 2

        I = CriticalPrice(id_, I1, maturity1, maturity2, v, quantity_on_asset2)
        d1 = (np.log(underlying_price1 / (I * underlying_price2)) + (
                carry_cost1 - carry_cost2 + v ** 2 / 2) * maturity1) / (v * np.sqrt(maturity1))
        d2 = d1 - v * np.sqrt(maturity1)
        d3 = (np.log((I * underlying_price2) / underlying_price1) + (
                carry_cost2 - carry_cost1 + v ** 2 / 2) * maturity1) / (v * np.sqrt(maturity1))
        d4 = d3 - v * np.sqrt(maturity1)
        y1 = (np.log(underlying_price1 / underlying_price2) + (
                carry_cost1 - carry_cost2 + v ** 2 / 2) * maturity2) / (
                     v * np.sqrt(maturity2))
        y2 = y1 - v * np.sqrt(maturity2)
        y3 = (np.log(underlying_price2 / underlying_price1) + (
                carry_cost2 - carry_cost1 + v ** 2 / 2) * maturity2) / (
                     v * np.sqrt(maturity2))
        y4 = y3 - v * np.sqrt(maturity2)

        if int(options_type) == 1:
            return -underlying_price2 * np.exp((carry_cost2 - rate) * maturity2) * CBND(d2, y2, np.sqrt(
                maturity1 / maturity2)) + underlying_price1 * np.exp((carry_cost1 - rate) * maturity2) * \
                CBND(d1, y1, np.sqrt(maturity1 / maturity2)) - quantity_on_asset2 * underlying_price2 * np.exp(
                    (carry_cost2 - rate) * maturity1) * norm.cdf(d2)
        elif int(options_type) == 2:
            return underlying_price2 * np.exp((carry_cost2 - rate) * maturity2) * CBND(d3, y2, -np.sqrt(
                maturity1 / maturity2)) - underlying_price1 * np.exp((carry_cost1 - rate) * maturity2) * \
                CBND(d4, y1, -np.sqrt(maturity1 / maturity2)) + quantity_on_asset2 * underlying_price2 * np.exp(
                    (carry_cost2 - rate) * maturity1) * norm.cdf(d3)
        elif int(options_type) == 3:
            return underlying_price2 * np.exp((carry_cost2 - rate) * maturity2) * CBND(d3, y3, np.sqrt(
                maturity1 / maturity2)) - underlying_price1 * np.exp((carry_cost1 - rate) * maturity2) * \
                CBND(d4, y4, np.sqrt(maturity1 / maturity2)) - quantity_on_asset2 * underlying_price2 * np.exp(
                    (carry_cost2 - rate) * maturity1) * norm.cdf(d3)
        elif int(options_type) == 4:
            return -underlying_price2 * np.exp((carry_cost2 - rate) * maturity2) * CBND(d2, y3, -np.sqrt(
                maturity1 / maturity2)) + underlying_price1 * np.exp((carry_cost1 - rate) * maturity2) * \
                CBND(d1, y4, -np.sqrt(maturity1 / maturity2)) + quantity_on_asset2 * underlying_price2 * np.exp(
                    (carry_cost2 - rate) * maturity1) * norm.cdf(d2)
        else:
            raise NotImplemented

    @staticmethod
    def Options_On_Max_Min_Risk_Assets(underlying_price1, underlying_price2, strike_price, maturity,
                                       rate, carry_cost1, carry_cost2, vol1, vol2, correlation, options_type):
        """
        options on 2 risk assets with max/min price
        :param underlying_price1: assset 1 price
        :param underlying_price2: asset 2 price
        :param strike_price: strike price
        :param maturity: maturity
        :param rate: risk-free rate
        :param carry_cost1: carry cost on asset 1
        :param carry_cost2: carry cost on asset 2
        :param vol1: volatility on asset 1
        :param vol2: volatikuty on asset 2
        :param correlation: correlation between 2 assets
        :param options_type: cmin/cmax/pmin/pmmax
        [1] Call on the minimum	cmin
        [2] Call on the maximum	cmax
        [3] Put on the minimum	pmin
        [4] Put on the maximum	pmax
        :return: options price
        """
        v = np.sqrt(vol1 ** 2 + vol2 ** 2 - 2 * correlation * vol1 * vol2)
        correlation1 = (vol1 - correlation * vol2) / v
        correlation2 = (vol2 - correlation * vol1) / v
        d = (np.log(underlying_price1 / underlying_price2) + (carry_cost1 - carry_cost2 + v ** 2 / 2) * maturity) / (
                v * np.sqrt(maturity))
        y1 = (np.log(underlying_price1 / strike_price) + (carry_cost1 + vol1 ** 2 / 2) * maturity) / (
                vol1 * np.sqrt(maturity))
        y2 = (np.log(underlying_price2 / strike_price) + (carry_cost2 + vol2 ** 2 / 2) * maturity) / (
                vol2 * np.sqrt(maturity))

        if options_type.lower() == 'cmin':
            return underlying_price1 * np.exp((carry_cost1 - rate) * maturity) * CBND(y1, -d, -correlation1) + \
                underlying_price2 * np.exp((carry_cost2 - rate) * maturity) * CBND(y2, d - v * np.sqrt(maturity),
                                                                                   -correlation2) - strike_price * \
                np.exp(-rate * maturity) * CBND(y1 - vol1 * np.sqrt(maturity), y2 - vol2 * np.sqrt(maturity),
                                                correlation)
        elif options_type.lower() == 'cmax':
            return underlying_price1 * np.exp((carry_cost1 - rate) * maturity) * CBND(y1, d,
                                                                                      correlation1) + underlying_price2 * np.exp(
                (carry_cost2 - rate) * maturity) * CBND(y2, -d + v * np.sqrt(maturity),
                                                        correlation2) - strike_price * np.exp(-rate * maturity) * (
                    1 - CBND(-y1 + vol1 * np.sqrt(maturity), -y2 + vol2 * np.sqrt(maturity), correlation))
        elif options_type.lower() == 'pmin':
            return strike_price * np.exp(-rate * maturity) - underlying_price1 * np.exp(
                (carry_cost1 - rate) * maturity) + EuropeanExchangeOption(underlying_price1, underlying_price2, 1, 1,
                                                                          maturity, rate, carry_cost1, carry_cost2,
                                                                          vol1, vol2, correlation) + \
                ExoticOPtions.Options_On_Max_Min_Risk_Assets(underlying_price1, underlying_price2, strike_price,
                                                             maturity, rate, carry_cost1, carry_cost2, vol1, vol2,
                                                             correlation, 'cmin')
        elif options_type.lower() == 'pmax':
            return strike_price * np.exp(-rate * maturity) - underlying_price2 * np.exp(
                (carry_cost2 - rate) * maturity) - EuropeanExchangeOption(underlying_price1, underlying_price2, 1, 1,
                                                                          maturity, rate, carry_cost1, carry_cost2,
                                                                          vol1, vol2, correlation) + \
                ExoticOPtions.Options_On_Max_Min_Risk_Assets(underlying_price1, underlying_price2, strike_price,
                                                             maturity, rate, carry_cost1, carry_cost2, vol1, vol2,
                                                             correlation, 'cmax')
        else:
            raise NotImplemented

    @staticmethod
    def Spread_Options(futures1, futures2, strike_price, maturity, rate, vol1, vol2, correlation, options_type):
        """
        spread options pricing model
        :param futures1: futures contract 1 price
        :param futures2: futures contract 2 price
        :param strike_price: options strike price
        :param maturity: maturity time
        :param rate: risk-free rate
        :param vol1: volatility on futures 1
        :param vol2: volatility on futures 2
        :param correlation: correlation between 2 futures
        :param options_type: call/put
        :return: options price
        """
        v = np.sqrt(
            vol1 ** 2 + (
                    vol2 * futures2 / (futures2 + strike_price)) ** 2 - 2 * correlation * vol1 * vol2 * futures2 / (
                    futures2 + strike_price))
        F = futures1 / (futures2 + strike_price)
        return BSM(F, 1, maturity, rate, 0, v, options_type) * (futures2 + strike_price)

    @staticmethod
    def Float_Strike_Lookback_Options(underlying_price, observed_min, observed_max, maturity, rate, carry_cost, vol,
                                      options_type):
        """
        float strike lookback options
        :param underlying_price: asset underlying price
        :param observed_min: observed periods min price
        :param observed_max: observed periods max price
        :param maturity: maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: opitons price
        """
        if options_type.lower() == 'call':
            strike_price = observed_min
        elif options_type.lower() == 'put':
            strike_price = observed_max
        else:
            raise TypeError
        a1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        a2 = a1 - vol * np.sqrt(maturity)

        if options_type.lower() == 'call':
            return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(a1) - strike_price * np.exp(
                -rate * maturity) * norm.cdf(a2) + \
                np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * underlying_price * (
                        (underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * norm.cdf(
                    -a1 + 2 * carry_cost / vol * np.sqrt(maturity)) - np.exp(carry_cost * maturity) * norm.cdf(-a1))
        elif options_type.lower() == 'put':
            return strike_price * np.exp(-rate * maturity) * norm.cdf(-a2) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(-a1) + \
                np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * underlying_price * (
                        -(underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * norm.cdf(
                    a1 - 2 * carry_cost / vol * np.sqrt(maturity)) + np.exp(carry_cost * maturity) * norm.cdf(a1))
        else:
            raise TypeError

    @staticmethod
    def Fixed_Strike_Lookback_Options(underlying_price, strike_price, observed_min, observed_max, maturity, rate,
                                      carry_cost, vol, options_type):
        """
        float strike lookback options
        :param underlying_price: asset underlying price
        :param strike_price: strike price on options
        :param observed_min: observed periods min price
        :param observed_max: observed periods max price
        :param maturity: maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: opitons price
        """
        if options_type.lower() == 'call':
            m = observed_max
        else:
            m = observed_min
        d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)
        e1 = (np.log(underlying_price / m) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
        e2 = e1 - vol * np.sqrt(maturity)

        if options_type.lower() == "call" and strike_price > m:
            return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) - strike_price * np.exp(
                -rate * maturity) * norm.cdf(d2) \
                + underlying_price * np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * (
                        -(underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * norm.cdf(
                    d1 - 2 * carry_cost / vol * np.sqrt(maturity)) + np.exp(carry_cost * maturity) * norm.cdf(d1))
        elif options_type.lower() == "call" and strike_price <= m:
            return np.exp(-rate * maturity) * (m - strike_price) + underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(e1) - np.exp(-rate * maturity) * m * norm.cdf(e2) \
                + underlying_price * np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * (
                        -(underlying_price / m) ** (-2 * carry_cost / vol ** 2) * norm.cdf(
                    e1 - 2 * carry_cost / vol * np.sqrt(maturity)) + np.exp(carry_cost * maturity) * norm.cdf(e1))
        elif options_type.lower() == "put" and strike_price < m:
            return -underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(-d1) + strike_price * np.exp(
                -rate * maturity) * norm.cdf(-d1 + vol * np.sqrt(maturity)) + underlying_price * np.exp(
                -rate * maturity) * vol ** 2 / (2 * carry_cost) * (
                    (underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * norm.cdf(
                -d1 + 2 * carry_cost / vol * np.sqrt(maturity)) - np.exp(carry_cost * maturity) * norm.cdf(-d1))
        elif options_type.lower() == "put" and strike_price >= m:
            return np.exp(-rate * maturity) * (strike_price - m) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(-e1) + np.exp(-rate * maturity) * m * norm.cdf(
                -e1 + vol * np.sqrt(maturity)) + np.exp(-rate * maturity) * vol ** 2 / (
                    2 * carry_cost) * underlying_price * (
                    (underlying_price / m) ** (-2 * carry_cost / vol ** 2) * norm.cdf(
                -e1 + 2 * carry_cost / vol * np.sqrt(maturity)) - np.exp(carry_cost * maturity) * norm.cdf(-e1))
        else:
            return None

    @staticmethod
    def Partial_Float_LookBack_Options(underlying_price, observed_min, observed_max, a_b_actual_extremum,
                                       lookback_length,
                                       maturity, rate, carry_cost, vol, options_type):
        """
        partial float lookback options
        :param underlying_price: asset underlying price
        :param observed_min: observed minimum price
        :param observed_max: observed maxmimum price
        :param a_b_actual_extremum: above or below actual extremum
        :param lookback_length: lookback time period length
        :param maturity: time to maturity
        :param rate: risk-free risk
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: options type
        :return: options price
        """
        if options_type.lower() in ['c', 'call']:
            m = observed_min
        else:
            m = observed_max
        d1 = (np.log(underlying_price / m) + (carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)
        e1 = (carry_cost + vol ** 2 / 2) * (maturity - lookback_length) / (vol * np.sqrt(maturity - lookback_length))
        e2 = e1 - vol * np.sqrt(maturity - lookback_length)
        f1 = (np.log(underlying_price / m) + (carry_cost + vol ** 2 / 2) * lookback_length) / (
                vol * np.sqrt(lookback_length))
        f2 = f1 - vol * np.sqrt(lookback_length)
        g1 = np.log(a_b_actual_extremum) / (vol * np.sqrt(maturity))
        g2 = np.log(a_b_actual_extremum) / (vol * np.sqrt(maturity - lookback_length))

        if options_type.lower() in ['c', 'call']:
            part1 = underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(
                d1 - g1) - a_b_actual_extremum * m * np.exp(-rate * maturity) * norm.cdf(d2 - g1)
            part2 = np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * \
                    a_b_actual_extremum * underlying_price * (
                            (underlying_price / m) ** (-2 * carry_cost / vol ** 2) * CBND(
                        -f1 + 2 * carry_cost * np.sqrt(lookback_length) / vol,
                        -d1 + 2 * carry_cost * np.sqrt(maturity) / vol - g1, np.sqrt(lookback_length / maturity)) -
                            np.exp(carry_cost * maturity) * a_b_actual_extremum ** (
                                    2 * carry_cost / vol ** 2) * CBND(-d1 - g1, e1 + g2, -np.sqrt(
                        1 - lookback_length / maturity))) + \
                    underlying_price * np.exp((carry_cost - rate) * maturity) * CBND(-d1 + g1, e1 - g2, -np.sqrt(
                1 - lookback_length / maturity))
            part3 = np.exp(-rate * maturity) * a_b_actual_extremum * m * CBND(-f2, d2 - g1, -np.sqrt(
                lookback_length / maturity)) - np.exp(-carry_cost * (maturity - lookback_length)) * np.exp(
                (carry_cost - rate) * maturity) * \
                    (1 + vol ** 2 / (2 * carry_cost)) * a_b_actual_extremum * underlying_price * norm.cdf(
                e2 - g2) * norm.cdf(-f1)

        else:
            part1 = a_b_actual_extremum * m * np.exp(-rate * maturity) * norm.cdf(-d2 + g1) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(-d1 + g1)
            part2 = -np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * a_b_actual_extremum * underlying_price * (
                    (underlying_price / m) ** (-2 * carry_cost / vol ** 2) * CBND(
                f1 - 2 * carry_cost * np.sqrt(lookback_length) /
                vol, d1 - 2 * carry_cost * np.sqrt(maturity) / vol + g1,
                np.sqrt(lookback_length / maturity)) - np.exp(carry_cost * maturity) * a_b_actual_extremum ** (
                            2 * carry_cost / vol ** 2) *
                    CBND(d1 + g1, -e1 - g2, -np.sqrt(1 - lookback_length / maturity))) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * CBND(d1 - g1, -e1 + g2, -np.sqrt(1 - lookback_length / maturity))
            part3 = -np.exp(-rate * maturity) * a_b_actual_extremum * m * CBND(f2, -d2 + g1, -np.sqrt(
                lookback_length / maturity)) + np.exp(-carry_cost * (maturity - lookback_length)) * np.exp(
                (carry_cost - rate) * maturity) * \
                    (1 + vol ** 2 / (2 * carry_cost)) * a_b_actual_extremum * underlying_price * norm.cdf(
                -e2 + g2) * norm.cdf(f1)

        return part1 + part2 + part3

    @staticmethod
    def Partial_Fixed_LookBack_Options(underlying_price, strike_price, lookback_length, maturity, rate, carry_cost, vol,
                                       options_type):
        """
        partial fixed lookback options
        :param underlying_price: asset underlying price
        :param strike_price: strike price on options
        :param lookback_length: lookback time period length
        :param maturity: time to maturity
        :param rate: risk-free risk
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: options type
        :return: options price
        """
        d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)
        e1 = ((carry_cost + vol ** 2 / 2) * (maturity - lookback_length)) / (vol * np.sqrt(maturity - lookback_length))
        e2 = e1 - vol * np.sqrt(maturity - lookback_length)
        f1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * lookback_length) / (
                vol * np.sqrt(lookback_length))
        f2 = f1 - vol * np.sqrt(lookback_length)

        if options_type.lower() == 'call':
            return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) - np.exp(
                -rate * maturity) * strike_price * norm.cdf(d2) + underlying_price * np.exp(
                -rate * maturity) * vol ** 2 / (2 * carry_cost) * (
                    -(underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * CBND(
                d1 - 2 * carry_cost * np.sqrt(maturity) / vol,
                -f1 + 2 * carry_cost * np.sqrt(lookback_length) / vol,
                -np.sqrt(lookback_length / maturity)) + np.exp(carry_cost * maturity) * CBND(e1, d1, np.sqrt(
                1 - lookback_length / maturity))) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * CBND(-e1, d1, -np.sqrt(
                1 - lookback_length / maturity)) - strike_price * np.exp(-rate * maturity) * CBND(f2, -d2, -np.sqrt(
                lookback_length / maturity)) + np.exp(-carry_cost * (maturity - lookback_length)) * (
                    1 - vol ** 2 / (2 * carry_cost)) * underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(f1) * norm.cdf(-e2)
        else:
            return strike_price * np.exp(-rate * maturity) * norm.cdf(-d2) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(-d1) + underlying_price * np.exp(
                -rate * maturity) * vol ** 2 / (2 * carry_cost) * (
                    (underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * CBND(
                -d1 + 2 * carry_cost * np.sqrt(maturity) / vol,
                f1 - 2 * carry_cost * np.sqrt(lookback_length) / vol,
                -np.sqrt(lookback_length / maturity)) - np.exp(carry_cost * maturity) * CBND(-e1, -d1, np.sqrt(
                1 - lookback_length / maturity))) + underlying_price * np.exp(
                (carry_cost - rate) * maturity) * CBND(e1, -d1, -np.sqrt(
                1 - lookback_length / maturity)) + strike_price * np.exp(-rate * maturity) * CBND(-f2, d2, -np.sqrt(
                lookback_length / maturity)) - np.exp(-carry_cost * (maturity - lookback_length)) * (
                    1 - vol ** 2 / (2 * carry_cost)) * underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(-f1) * norm.cdf(e2)

    @staticmethod
    def Extreme_Spread_Options(underlying_price, observed_min, observed_max, maturity1, maturity, rate, carry_cost, vol,
                               options_type):
        """
        partial fixed lookback options
        :param underlying_price: asset underlying price
        :param observed_min: minimum observed price
        :param observed_max: maximum observed priceength
        :param maturity1: first time priod
        :param maturity: time to maturity
        :param rate: risk-free risk
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: user must use 1~4 number as input
        :return: options price
        """
        if int(options_type) in [1, 3]:
            eta = 1
        else:
            eta = -1

        if int(options_type) in [1, 2]:
            kapper = 1
        else:
            kapper = -1

        if eta * kapper == 1:
            Mo = observed_max
        else:
            Mo = observed_min

        mu1 = carry_cost - vol ** 2 / 2
        mu = mu1 + vol ** 2
        m = np.log(Mo / underlying_price)

        if kapper == 1:
            return eta * (underlying_price * np.exp((carry_cost - rate) * maturity) * (
                    1 + vol ** 2 / (2 * carry_cost)) * norm.cdf(
                eta * (-m + mu * maturity) / (vol * np.sqrt(maturity))) - np.exp(
                -rate * (maturity - maturity1)) * underlying_price * np.exp((carry_cost - rate) * maturity) * (
                                  1 + vol ** 2 / (2 * carry_cost)) * norm.cdf(
                eta * (-m + mu * maturity1) / (vol * np.sqrt(maturity1))) + np.exp(-rate * maturity) * Mo * norm.cdf(
                eta * (m - mu1 * maturity) / (vol * np.sqrt(maturity))) - np.exp(
                -rate * maturity) * Mo * vol ** 2 / (
                                  2 * carry_cost) * np.exp(2 * mu1 * m / vol ** 2) * norm.cdf(
                eta * (-m - mu1 * maturity) / (vol * np.sqrt(maturity))) - np.exp(
                -rate * maturity) * Mo * norm.cdf(eta * (m - mu1 * maturity1) / (vol * np.sqrt(maturity1))) + np.exp(
                -rate * maturity) * Mo * vol ** 2 / (
                                  2 * carry_cost) * np.exp(2 * mu1 * m / vol ** 2) * norm.cdf(
                eta * (-m - mu1 * maturity1) / (vol * np.sqrt(maturity1))))
        else:
            return -eta * (underlying_price * np.exp((carry_cost - rate) * maturity) * (
                    1 + vol ** 2 / (2 * carry_cost)) * norm.cdf(
                eta * (m - mu * maturity) / (vol * np.sqrt(maturity))) + np.exp(-rate * maturity) * Mo * norm.cdf(
                eta * (-m + mu1 * maturity) / (vol * np.sqrt(maturity))) - np.exp(-rate * maturity) * Mo * vol ** 2 / (
                                   2 * carry_cost) * np.exp(2 * mu1 * m / vol ** 2) * norm.cdf(
                eta * (m + mu1 * maturity) / (vol * np.sqrt(maturity))) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * (1 + vol ** 2 / (2 * carry_cost)) * norm.cdf(
                eta * (-mu * (maturity - maturity1)) / (vol * np.sqrt(maturity - maturity1))) - np.exp(
                -rate * (maturity - maturity1)) * underlying_price * np.exp((carry_cost - rate) * maturity) * (
                                   1 - vol ** 2 / (2 * carry_cost)) * norm.cdf(
                eta * (mu1 * (maturity - maturity1)) / (vol * np.sqrt(maturity - maturity1))))

    @staticmethod
    def Standard_Barrier_Options(underlying_price, strike_price, barrier, cash_rate, maturity, rate, carry_cost, vol,
                                 options_type):
        """
        standard barrier option
        :param underlying_price: asset underlying price
        :param barrier: barrier price for out or in
        :param cash_rate: price after reatching barrier price
        :param maturity: time to maturity
        :param rate: risk-free risk
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: user must use 1~4 number as input
        :return: options price
        """
        mu = (carry_cost - vol ** 2 / 2) / vol ** 2
        lambda_ = np.sqrt(mu ** 2 + 2 * rate / vol ** 2)
        strike_price1 = np.log(underlying_price / strike_price) / (vol * np.sqrt(maturity)) + (1 + mu) * vol * np.sqrt(
            maturity)
        strike_price2 = np.log(underlying_price / barrier) / (vol * np.sqrt(maturity)) + (1 + mu) * vol * np.sqrt(
            maturity)
        y1 = np.log(barrier ** 2 / (underlying_price * strike_price)) / (vol * np.sqrt(maturity)) + (
                1 + mu) * vol * np.sqrt(maturity)
        y2 = np.log(barrier / underlying_price) / (vol * np.sqrt(maturity)) + (1 + mu) * vol * np.sqrt(maturity)
        Z = np.log(barrier / underlying_price) / (vol * np.sqrt(maturity)) + lambda_ * vol * np.sqrt(maturity)
        if options_type.lower() in ['cdi', 'cdo']:
            eta = 1
            phi = 1
        elif options_type.lower() in ['cui', 'cuo']:
            eta = -1
            phi = 1
        elif options_type.lower() in ['pdi', 'pdo']:
            eta = 1
            phi = -1
        elif options_type.lower() in ['pui', 'puo']:
            eta = -1
            phi = -1
        else:
            raise TypeError

        f1 = phi * underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(
            phi * strike_price1) - phi * strike_price * np.exp(-rate * maturity) * norm.cdf(
            phi * strike_price1 - phi * vol * np.sqrt(maturity))
        f2 = phi * underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(
            phi * strike_price2) - phi * strike_price * np.exp(-rate * maturity) * norm.cdf(
            phi * strike_price2 - phi * vol * np.sqrt(maturity))
        f3 = phi * underlying_price * np.exp((carry_cost - rate) * maturity) * (barrier / underlying_price) ** (
                2 * (mu + 1)) * norm.cdf(eta * y1) - phi * strike_price * np.exp(-rate * maturity) * (
                     barrier / underlying_price) ** (
                     2 * mu) * norm.cdf(eta * y1 - eta * vol * np.sqrt(maturity))
        f4 = phi * underlying_price * np.exp((carry_cost - rate) * maturity) * (barrier / underlying_price) ** (
                2 * (mu + 1)) * norm.cdf(eta * y2) - phi * strike_price * np.exp(-rate * maturity) * (
                     barrier / underlying_price) ** (
                     2 * mu) * norm.cdf(eta * y2 - eta * vol * np.sqrt(maturity))
        f5 = cash_rate * np.exp(-rate * maturity) * (
                norm.cdf(eta * strike_price2 - eta * vol * np.sqrt(maturity)) - (barrier / underlying_price) ** (
                2 * mu) * norm.cdf(eta * y2 - eta * vol * np.sqrt(maturity)))
        f6 = cash_rate * ((barrier / underlying_price) ** (mu + lambda_) * norm.cdf(eta * Z) + (
                barrier / underlying_price) ** (mu - lambda_) * norm.cdf(eta * Z - 2 * eta *
                                                                         lambda_ * vol * np.sqrt(maturity)))

        if strike_price > barrier:
            if options_type.lower() == 'cdi':
                return f3 + f5
            elif options_type.lower() == 'cui':
                return f1 + f5
            elif options_type.lower() == 'pdi':
                return f2 - f3 + f4 + f5
            elif options_type.lower() == 'pui':
                return f1 - f2 + f4 + f5
            elif options_type.lower() == 'cdo':
                return f1 - f3 + f6
            elif options_type.lower() == 'cuo':
                return f6
            elif options_type.lower() == 'pdo':
                return f1 - f2 + f3 - f4 + f6
            elif options_type.lower() == 'puo':
                return f2 - f4 + f6
            else:
                raise TypeError
        elif barrier > strike_price:
            if options_type.lower() == 'cdi':
                return f1 - f2 + f4 + f5
            elif options_type.lower() == 'cui':
                return f2 - f3 + f4 + f5
            elif options_type.lower() == 'pdi':
                return f1 + f5
            elif options_type.lower() == 'pui':
                return f3 + f5
            elif options_type.lower() == 'cdo':
                return f2 + f6 - f4
            elif options_type.lower() == 'cuo':
                return f1 - f2 + f3 - f4 + f6
            elif options_type.lower() == 'pdo':
                return f6
            elif options_type.lower() == 'puo':
                return f1 - f3 + f6
            else:
                raise TypeError

    @staticmethod
    def Double_Barrier_Options(underlying_price, strike_price, low_barrier, high_barrier, maturity, rate,
                               carry_cost, vol, options_type, curvature_upper, curvature_lower):
        """
        double barrier option
        :param underlying_price: asset underlying price
        :param low_barrier: lower barrier price for out or in
        :param high_barrier: higher barrier price for out or in
        :param cash_rate: price after reatching barrier price
        :param maturity: time to maturity
        :param rate: risk-free risk
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: user must use 1~4 number as input
        :param curvature_upper: uppder curvature for double barrier options
        :param curvature_bottom: lower curvature for double barrier options
        :return: options price
        """
        F = high_barrier * np.exp(curvature_upper * maturity)
        E = low_barrier * np.exp(curvature_upper * maturity)
        Sum1 = 0
        Sum2 = 0

        if options_type.lower() in ['co', 'ci']:
            for n in range(-5, 6):
                d1 = (np.log(underlying_price * high_barrier ** (2 * n) / (strike_price * low_barrier ** (2 * n))) + (
                        carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                d2 = (np.log(underlying_price * high_barrier ** (2 * n) / (F * low_barrier ** (2 * n))) + (
                        carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                d3 = (np.log(
                    low_barrier ** (2 * n + 2) / (strike_price * underlying_price * high_barrier ** (2 * n))) + (
                              carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                d4 = (np.log(low_barrier ** (2 * n + 2) / (F * underlying_price * high_barrier ** (2 * n))) + (
                        carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                mu1 = 2 * (carry_cost - curvature_lower - n * (curvature_upper - curvature_lower)) / vol ** 2 + 1
                mu2 = 2 * n * (curvature_upper - curvature_lower) / vol ** 2
                mu3 = 2 * (carry_cost - curvature_lower + n * (curvature_upper - curvature_lower)) / vol ** 2 + 1
                Sum1 = Sum1 + (high_barrier ** n / low_barrier ** n) ** mu1 * (
                        low_barrier / underlying_price) ** mu2 * (norm.cdf(d1) - norm.cdf(d2)) - (
                               low_barrier ** (n + 1) / (high_barrier ** n * underlying_price)) ** mu3 * (
                               norm.cdf(d3) - norm.cdf(d4))
                Sum2 = Sum2 + (high_barrier ** n / low_barrier ** n) ** (mu1 - 2) * (
                        low_barrier / underlying_price) ** mu2 * (
                               norm.cdf(d1 - vol * np.sqrt(maturity)) - norm.cdf(d2 - vol * np.sqrt(maturity))) - (
                               low_barrier ** (n + 1) / (high_barrier ** n * underlying_price)) ** (mu3 - 2) * (
                               norm.cdf(d3 - vol * np.sqrt(maturity)) - norm.cdf(d4 - vol * np.sqrt(maturity)))

            OutValue = underlying_price * np.exp((carry_cost - rate) * maturity) * Sum1 - strike_price * np.exp(
                -rate * maturity) * Sum2
        elif options_type.lower() in ['po', 'pi']:
            for n in range(-5, 6):
                d1 = (np.log(underlying_price * high_barrier ** (2 * n) / (E * low_barrier ** (2 * n))) + (
                        carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                d2 = (np.log(underlying_price * high_barrier ** (2 * n) / (strike_price * low_barrier ** (2 * n))) + (
                        carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                d3 = (np.log(low_barrier ** (2 * n + 2) / (E * underlying_price * high_barrier ** (2 * n))) + (
                        carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                d4 = (np.log(
                    low_barrier ** (2 * n + 2) / (strike_price * underlying_price * high_barrier ** (2 * n))) + (
                              carry_cost + vol ** 2 / 2) * maturity) / (vol * np.sqrt(maturity))
                mu1 = 2 * (carry_cost - curvature_lower - n * (curvature_upper - curvature_lower)) / vol ** 2 + 1
                mu2 = 2 * n * (curvature_upper - curvature_lower) / vol ** 2
                mu3 = 2 * (carry_cost - curvature_lower + n * (curvature_upper - curvature_lower)) / vol ** 2 + 1
                Sum1 = Sum1 + (high_barrier ** n / low_barrier ** n) ** mu1 * (
                        low_barrier / underlying_price) ** mu2 * (norm.cdf(d1) - norm.cdf(d2)) - (
                               low_barrier ** (n + 1) / (high_barrier ** n * underlying_price)) ** mu3 * (
                               norm.cdf(d3) - norm.cdf(d4))
                Sum2 = Sum2 + (high_barrier ** n / low_barrier ** n) ** (mu1 - 2) * (
                        low_barrier / underlying_price) ** mu2 * (
                               norm.cdf(d1 - vol * np.sqrt(maturity)) - norm.cdf(d2 - vol * np.sqrt(maturity))) - (
                               low_barrier ** (n + 1) / (high_barrier ** n * underlying_price)) ** (mu3 - 2) * (
                               norm.cdf(d3 - vol * np.sqrt(maturity)) - norm.cdf(d4 - vol * np.sqrt(maturity)))
            OutValue = strike_price * np.exp(-rate * maturity) * Sum2 - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * Sum1
        else:
            raise TypeError('only supporting "co/ci/po/pi"')

        if options_type.lower() in ['co', 'po']:
            return OutValue
        elif options_type.lower() == 'ci':
            return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, 'call') - OutValue
        elif options_type.lower() == 'pi':
            return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, 'put') - OutValue
        else:
            raise TypeError

    @staticmethod
    def Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1, maturity2, rate,
                                carry_cost, vol, options_type):
        """
        partial barrier option
        :param underlying_price: asset underlying price
        :param barrier:barrier price for out or in
        :param cash_rate: price after reatching barrier price
        :param maturity: time to maturity
        :param rate: risk-free risk
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: user must use 1~4 number as input
        :return: options price
        """
        if options_type.lower() == 'cdoa':
            eta = 1
        elif options_type.lower() == 'cuoa':
            eta = -1
        d1 = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity2) / (
                vol * np.sqrt(maturity2))
        d2 = d1 - vol * np.sqrt(maturity2)
        f1 = (np.log(underlying_price / strike_price) + 2 * np.log(barrier / underlying_price) + (
                carry_cost + vol ** 2 / 2) * maturity2) / (vol * np.sqrt(maturity2))
        f2 = f1 - vol * np.sqrt(maturity2)
        e1 = (np.log(underlying_price / barrier) + (carry_cost + vol ** 2 / 2) * maturity1) / (vol * np.sqrt(maturity1))
        e2 = e1 - vol * np.sqrt(maturity1)
        e3 = e1 + 2 * np.log(barrier / underlying_price) / (vol * np.sqrt(maturity1))
        e4 = e3 - vol * np.sqrt(maturity1)
        mu = (carry_cost - vol ** 2 / 2) / vol ** 2
        corr_ = np.sqrt(maturity1 / maturity2)
        g1 = (np.log(underlying_price / barrier) + (carry_cost + vol ** 2 / 2) * maturity2) / (vol * np.sqrt(maturity2))
        g2 = g1 - vol * np.sqrt(maturity2)
        g3 = g1 + 2 * np.log(barrier / underlying_price) / (vol * np.sqrt(maturity2))
        g4 = g3 - vol * np.sqrt(maturity2)

        z1 = norm.cdf(e2) - (barrier / underlying_price) ** (2 * mu) * norm.cdf(e4)
        z2 = norm.cdf(-e2) - (barrier / underlying_price) ** (2 * mu) * norm.cdf(-e4)
        z3 = CBND(g2, e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(g4, -e4, -corr_)
        z4 = CBND(-g2, -e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(-g4, e4, -corr_)
        z5 = norm.cdf(e1) - (barrier / underlying_price) ** (2 * (mu + 1)) * norm.cdf(e3)
        z6 = norm.cdf(-e1) - (barrier / underlying_price) ** (2 * (mu + 1)) * norm.cdf(-e3)
        z7 = CBND(g1, e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(g3, -e3, -corr_)
        z8 = CBND(-g1, -e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(-g3, e3, -corr_)

        if options_type.lower() in ['cdoa', 'cuoa']:
            result = underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                    CBND(d1, eta * e1, eta * corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(f1,
                                                                                                            eta * e3,
                                                                                                            eta * corr_)) - \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(d2, eta * e2, eta * corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(f2,
                                                                                                               eta * e4,
                                                                                                               eta * corr_))
        elif options_type.lower() == 'cdob2' and strike_price < barrier:
            result = underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                    CBND(g1, e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(g3, -e3, -corr_)) - \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(g2, e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(g4, -e4, -corr_))
        elif options_type.lower() == 'cdob2' and strike_price > barrier:
            result = ExoticOPtions.Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1,
                                                           maturity2, rate, carry_cost, vol, "coB1")
        elif options_type.lower() == 'cuob2' and strike_price < barrier:
            result = underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                    CBND(-g1, -e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(-g3, e3, -corr_)) - \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(-g2, -e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(-g4, e4, -corr_)) - \
                     underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                             CBND(-d1, -e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(e3, -f1,
                                                                                                           -corr_)) + \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(-d2, -e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(e4, -f2, -corr_))
        elif options_type.lower() == 'cob1' and strike_price > barrier:
            result = underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                    CBND(d1, e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(f1, -e3, -corr_)) - \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(d2, e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(f2, -e4, -corr_))
        elif options_type.lower() == 'cob1' and strike_price < barrier:
            result = underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                    CBND(-g1, -e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(-g3, e3, -corr_)) - \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(-g2, -e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(-g4, e4, -corr_)) - \
                     underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                             CBND(-d1, -e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(-f1, e3,
                                                                                                           -corr_)) + \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(-d2, -e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(-f2, e4, -corr_)) + \
                     underlying_price * np.exp((carry_cost - rate) * maturity2) * (
                             CBND(g1, e1, corr_) - (barrier / underlying_price) ** (2 * (mu + 1)) * CBND(g3, -e3,
                                                                                                         -corr_)) - \
                     strike_price * np.exp(-rate * maturity2) * (
                             CBND(g2, e2, corr_) - (barrier / underlying_price) ** (2 * mu) * CBND(g4, -e4, -corr_))
        elif options_type.lower() == 'pdoa':
            result = ExoticOPtions.Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1,
                                                           maturity2, rate, carry_cost, vol,
                                                           "cdoA") - underlying_price * np.exp(
                (carry_cost - rate) * maturity2) * z5 + strike_price * np.exp(-rate * maturity2) * z1
        elif options_type.lower() == 'puoa':
            result = ExoticOPtions.Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1,
                                                           maturity2, rate, carry_cost, vol,
                                                           "cuoA") - underlying_price * np.exp(
                (carry_cost - rate) * maturity2) * z6 + strike_price * np.exp(-rate * maturity2) * z2
        elif options_type.lower() == 'pob1':
            result = ExoticOPtions.Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1,
                                                           maturity2, rate, carry_cost, vol,
                                                           "coB1") - underlying_price * np.exp(
                (carry_cost - rate) * maturity2) * z8 + strike_price * np.exp(
                -rate * maturity2) * z4 - underlying_price * np.exp(
                (carry_cost - rate) * maturity2) * z7 + strike_price * np.exp(-rate * maturity2) * z3
        elif options_type.lower() == 'pdob2':
            result = ExoticOPtions.Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1,
                                                           maturity2, rate, carry_cost, vol,
                                                           "cdoB2") - underlying_price * np.exp(
                (carry_cost - rate) * maturity2) * z7 + strike_price * np.exp(-rate * maturity2) * z3
        elif options_type.lower() == 'puob2':
            result = ExoticOPtions.Partial_Barrier_Options(underlying_price, strike_price, barrier, maturity1,
                                                           maturity2, rate, carry_cost, vol,
                                                           "cuoB2") - underlying_price * np.exp(
                (carry_cost - rate) * maturity2) * z8 + strike_price * np.exp(-rate * maturity2) * z4
        else:
            raise TypeError

        return result

    @staticmethod
    def maturitywo_Asets_Barrier_Options(underlying_price1, underlying_price2, strike_price, barrier, maturity,
                                         rate, carry_cost1, carry_cost2, vol1, vol2, corr_, options_type):
        """
        2 assets barrier options
        :param underlying_price1: price on asset1
        :param underlying_price2: price on asset2
        :param strike_price: strike price on 2 assets barrier options
        :param barrier: barrier price
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost1: carry cost on asset1
        :param carry_cost2: carry cost on asset2
        :param vol1: volatility on asset1
        :param vol2: volatility on asset2
        :param corr_: correlation between asset1 and asset2
        :param options_type: options type
        :return: option price
        """

        mu1 = carry_cost1 - vol1 ** 2 / 2
        mu2 = carry_cost2 - vol2 ** 2 / 2

        d1 = (np.log(underlying_price1 / strike_price) + (mu1 + vol1 ** 2 / 2) * maturity) / (vol1 * np.sqrt(maturity))
        d2 = d1 - vol1 * np.sqrt(maturity)
        d3 = d1 + 2 * corr_ * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(maturity))
        d4 = d2 + 2 * corr_ * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(maturity))
        e1 = (np.log(barrier / underlying_price2) - (mu2 + corr_ * vol1 * vol2) * maturity) / (vol2 * np.sqrt(maturity))
        e2 = e1 + corr_ * vol1 * np.sqrt(maturity)
        e3 = e1 - 2 * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(maturity))
        e4 = e2 - 2 * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(maturity))

        if options_type.lower() in ['cuo', 'cui']:
            eta = 1
            phi = 1
        elif options_type.lower() in ['cdo', 'cdi']:
            eta = 1
            phi = -1
        elif options_type.lower() in ['puo', 'pui']:
            eta = -1
            phi = 1
        elif options_type.lower() in ['pdo', 'pdi']:
            eta = -1
            phi = -1
        else:
            raise TypeError

        knock_out_value = eta * underlying_price1 * np.exp((carry_cost1 - rate) * maturity) * (
                CBND(eta * d1, phi * e1, -eta * phi * corr_) - \
                np.exp(2 * (mu2 + corr_ * vol1 * vol2) * np.log(barrier / underlying_price2) / vol2 ** 2) * CBND(
            eta * d3, phi * e3, -eta * phi * corr_)) - eta * np.exp(-rate * maturity) * strike_price * (
                                  CBND(eta * d2, phi * e2, -eta * phi * corr_) - \
                                  np.exp(2 * mu2 * np.log(barrier / underlying_price2) / vol2 ** 2) * CBND(eta * d4,
                                                                                                           phi * e4,
                                                                                                           -eta * phi * corr_))
        if options_type.lower() in ['cuo', 'cdo', 'puo', 'pdo']:
            return knock_out_value
        elif options_type.lower() in ['cui', 'cdi']:
            return BSM(underlying_price1, strike_price, maturity, rate, carry_cost1, vol1, 'call') - knock_out_value
        elif options_type.lower() in ['pui', 'pdi']:
            return BSM(underlying_price1, strike_price, maturity, rate, carry_cost1, vol1, 'put') - knock_out_value
        else:
            raise TypeError

    @staticmethod
    def Partial_maturitywo_Asets_Barrier_Options(underlying_price1, underlying_price2, strike_price, barrier,
                                                 barrier_holding_life, maturity, rate, carry_cost1, carry_cost2,
                                                 vol1, vol2, corr_, options_type):
        """
        2 assets barrier options
        :param underlying_price1: price on asset1
        :param underlying_price2: price on asset2
        :param strike_price: strike price on 2 assets barrier options
        :param barrier: barrier price
        :param barrier_holding_life: barrier holding lifeprice
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost1: carry cost on asset1
        :param carry_cost2: carry cost on asset2
        :param vol1: volatility on asset1
        :param vol2: volatility on asset2
        :param corr_: correlation between asset1 and asset2
        :param options_type: options type
        :return: option price
        """
        if options_type.lower() in ['cdo', 'pdo', 'cdi', 'pdi']:
            phi = -1
        else:
            phi = 1

        if options_type.lower() in ['cuo', 'cdo', 'cdi', 'cui']:
            eta = 1
        else:
            eta = -1
        mu1 = carry_cost1 - vol1 ** 2 / 2
        mu2 = carry_cost2 - vol2 ** 2 / 2
        d1 = (np.log(underlying_price1 / strike_price) + (mu1 + vol1 ** 2) * maturity) / (vol1 * np.sqrt(maturity))
        d2 = d1 - vol1 * np.sqrt(maturity)
        d3 = d1 + 2 * corr_ * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(maturity))
        d4 = d2 + 2 * corr_ * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(maturity))
        e1 = (np.log(barrier / underlying_price2) - (mu2 + corr_ * vol1 * vol2) * barrier_holding_life) / (
                vol2 * np.sqrt(barrier_holding_life))
        e2 = e1 + corr_ * vol1 * np.sqrt(barrier_holding_life)
        e3 = e1 - 2 * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(barrier_holding_life))
        e4 = e2 - 2 * np.log(barrier / underlying_price2) / (vol2 * np.sqrt(barrier_holding_life))

        OutBarrierValue = eta * underlying_price1 * np.exp((carry_cost1 - rate) * maturity) * (
                CBND(eta * d1, phi * e1, -eta * phi * corr_ * np.sqrt(barrier_holding_life / maturity)) - np.exp(
            2 * np.log(barrier / underlying_price2) * (mu2 + corr_ * vol1 * vol2) / (vol2 ** 2)) \
                * CBND(eta * d3, phi * e3, -eta * phi * corr_ * np.sqrt(barrier_holding_life / maturity))) \
                          - eta * np.exp(-rate * maturity) * strike_price * (
                                  CBND(eta * d2, phi * e2,
                                       -eta * phi * corr_ * np.sqrt(barrier_holding_life / maturity)) - np.exp(
                              2 * np.log(barrier / underlying_price2) * mu2 / (vol2 ** 2)) \
                                  * CBND(eta * d4, phi * e4,
                                         -eta * phi * corr_ * np.sqrt(barrier_holding_life / maturity)))

        if options_type.lower() in ['cdo', 'cuo', 'pdo', 'puo']:
            return OutBarrierValue
        elif options_type.lower() in ['cui', 'cdi']:
            return BSM(underlying_price1, strike_price, maturity, rate, carry_cost1, vol1, 'call') - OutBarrierValue
        elif options_type.lower() in ['pui', 'pdi']:
            return BSM(underlying_price1, strike_price, maturity, rate, carry_cost1, vol1, 'put') - OutBarrierValue
        else:
            raise TypeError

    @staticmethod
    def Lookback_Barrier_Options(underlying_price, strike_price, barrier, barrier_holding_life, maturity, rate,
                                 carry_cost, vol, options_type):
        """
        lookback barrier options
        :param underlying_price: price on underlying
        :param strike_price: strike price on options
        :param barrier: barrier price
        :param barrier_holding_life: barrier_holding_life
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        hh = np.log(barrier / underlying_price)
        K = np.log(strike_price / underlying_price)
        mu1 = carry_cost - vol ** 2 / 2
        mu2 = carry_cost + vol ** 2 / 2
        rho = np.sqrt(barrier_holding_life / maturity)

        if options_type.lower() in ['cuo', 'cui']:
            eta = 1
            m = min(hh, K)
        elif options_type.lower() in ['pdo', 'pdi']:
            eta = -1
            m = max(hh, K)
        else:
            raise TypeError
        g1 = (norm.cdf(eta * (hh - mu2 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life))) - np.exp(
            2 * mu2 * hh / vol ** 2) * norm.cdf(
            eta * (-hh - mu2 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)))) \
             - (norm.cdf(eta * (m - mu2 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life))) - np.exp(
            2 * mu2 * hh / vol ** 2) * norm.cdf(
            eta * (m - 2 * hh - mu2 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life))))

        g2 = (norm.cdf(eta * (hh - mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life))) - np.exp(
            2 * mu1 * hh / vol ** 2) * norm.cdf(
            eta * (-hh - mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)))) \
             - (norm.cdf(eta * (m - mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life))) - np.exp(
            2 * mu1 * hh / vol ** 2) * norm.cdf(
            eta * (m - 2 * hh - mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life))))

        part1 = underlying_price * np.exp((carry_cost - rate) * maturity) * (1 + vol ** 2 / (2 * carry_cost)) * (
                CBND(eta * (m - mu2 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)),
                     eta * (-K + mu2 * maturity) / (vol * np.sqrt(maturity)), -rho) - np.exp(
            2 * mu2 * hh / vol ** 2) \
                * CBND(eta * (m - 2 * hh - mu2 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)),
                       eta * (2 * hh - K + mu2 * maturity) / (vol * np.sqrt(maturity)), -rho))
        part2 = -np.exp(-rate * maturity) * strike_price * (
                CBND(eta * (m - mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)),
                     eta * (-K + mu1 * maturity) / (vol * np.sqrt(maturity)), -rho) \
                - np.exp(2 * mu1 * hh / vol ** 2) * CBND(
            eta * (m - 2 * hh - mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)),
            eta * (2 * hh - K + mu1 * maturity) / (vol * np.sqrt(maturity)), -rho))
        part3 = -np.exp(-rate * maturity) * vol ** 2 / (2 * carry_cost) * (
                underlying_price * (underlying_price / strike_price) ** (-2 * carry_cost / vol ** 2) * CBND(
            eta * (m + mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)),
            eta * (-K - mu1 * maturity) / (vol * np.sqrt(maturity)),
            -rho) \
                - barrier * (barrier / strike_price) ** (-2 * carry_cost / vol ** 2) * CBND(
            eta * (m - 2 * hh + mu1 * barrier_holding_life) / (vol * np.sqrt(barrier_holding_life)),
            eta * (2 * hh - K - mu1 * maturity) / (vol * np.sqrt(maturity)), -rho))
        part4 = underlying_price * np.exp((carry_cost - rate) * maturity) * (
                (1 + vol ** 2 / (2 * carry_cost)) * norm.cdf(
            eta * mu2 * (maturity - barrier_holding_life) / (vol * np.sqrt(maturity - barrier_holding_life))) + np.exp(
            -carry_cost * (maturity - barrier_holding_life)) * (
                        1 - vol ** 2 / (2 * carry_cost)) \
                * norm.cdf(eta * (-mu1 * (maturity - barrier_holding_life)) / (
                vol * np.sqrt(maturity - barrier_holding_life)))) * g1 - np.exp(-rate * maturity) * strike_price * g2
        OutValue = eta * (part1 + part2 + part3 + part4)

        if options_type.lower() in ['cuo', 'pdo']:
            return OutValue
        elif options_type.lower() in ['cui']:
            return ExoticOPtions.Partial_Fixed_LookBack_Options(underlying_price, strike_price, barrier_holding_life,
                                                                maturity, rate, carry_cost, vol, 'call') - OutValue
        elif options_type.lower() in ['pdi']:
            return ExoticOPtions.Partial_Fixed_LookBack_Options(underlying_price, strike_price, barrier_holding_life,
                                                                maturity, rate, carry_cost, vol, 'put') - OutValue
        else:
            raise TypeError

    @staticmethod
    def Soft_Barrier_Options(underlying_price, strike_price, lower_barrier, higher_barrier, maturity, rate,
                             carry_cost, vol, options_type):
        """
        soft barrier options
        :param underlying_price: price on underlying
        :param strike_price: strike price on options
        :param lower_barrier: lower barrier price
        :param hight_barrier: higher barrier price
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        if options_type.lower() in ['cdi', 'cdo']:
            eta = 1
        else:
            eta = -1
        mu = (carry_cost + vol ** 2 / 2) / vol ** 2
        lambda1 = np.exp(-1 / 2 * vol ** 2 * maturity * (mu + 0.5) * (mu - 0.5))
        lambda2 = np.exp(-1 / 2 * vol ** 2 * maturity * (mu - 0.5) * (mu - 1.5))
        d1 = np.log(higher_barrier ** 2 / (underlying_price * strike_price)) / (
                vol * np.sqrt(maturity)) + mu * vol * np.sqrt(maturity)
        d2 = d1 - (mu + 0.5) * vol * np.sqrt(maturity)
        d3 = np.log(higher_barrier ** 2 / (underlying_price * strike_price)) / (vol * np.sqrt(maturity)) + (
                mu - 1) * vol * np.sqrt(maturity)
        d4 = d3 - (mu - 0.5) * vol * np.sqrt(maturity)
        e1 = np.log(lower_barrier ** 2 / (underlying_price * strike_price)) / (
                vol * np.sqrt(maturity)) + mu * vol * np.sqrt(
            maturity)
        e2 = e1 - (mu + 0.5) * vol * np.sqrt(maturity)
        e3 = np.log(lower_barrier ** 2 / (underlying_price * strike_price)) / (vol * np.sqrt(maturity)) + (
                mu - 1) * vol * np.sqrt(maturity)
        e4 = e3 - (mu - 0.5) * vol * np.sqrt(maturity)

        Value = eta * 1 / (higher_barrier - lower_barrier) * (
                underlying_price * np.exp((carry_cost - rate) * maturity) * underlying_price ** (-2 * mu) \
                * (underlying_price * strike_price) ** (mu + 0.5) / (2 * (mu + 0.5)) \
                * ((higher_barrier ** 2 / (underlying_price * strike_price)) ** (mu + 0.5) * norm.cdf(
            eta * d1) - lambda1 * norm.cdf(eta * d2) \
                   - (lower_barrier ** 2 / (underlying_price * strike_price)) ** (mu + 0.5) * norm.cdf(
                    eta * e1) + lambda1 * norm.cdf(eta * e2)) \
                - strike_price * np.exp(-rate * maturity) * underlying_price ** (
                        -2 * (mu - 1)) \
                * (underlying_price * strike_price) ** (mu - 0.5) / (2 * (mu - 0.5)) \
                * ((higher_barrier ** 2 / (underlying_price * strike_price)) ** (mu - 0.5) * norm.cdf(
            eta * d3) - lambda2 * norm.cdf(eta * d4) \
                   - (lower_barrier ** 2 / (underlying_price * strike_price)) ** (mu - 0.5) * norm.cdf(
                    eta * e3) + lambda2 * norm.cdf(eta * e4)))
        if options_type.lower() in ['cdi', 'pui']:
            return Value
        elif options_type.lower() == 'cdo':
            return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, 'call') - Value
        elif options_type.lower() == 'puo':
            return BSM(underlying_price, strike_price, maturity, rate, carry_cost, vol, 'put') - Value
        else:
            raise TypeError

    @staticmethod
    def Gap_Options(underlying_price, strike_price1, strike_price2, maturity, rate, carry_cost, vol, options_type):
        """
        gap options price
        :param underlying_price: asset price
        :param strike_price1: strike price on asset1
        :param strike_price2: strike price on asset2
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        d1 = (np.log(underlying_price / strike_price1) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)
        if options_type.lower() == 'call':
            return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d1) - strike_price2 * np.exp(
                -rate * maturity) * norm.cdf(d2)
        elif options_type.lower() == 'put':
            return strike_price2 * np.exp(-rate * maturity) * norm.cdf(-d2) - underlying_price * np.exp(
                (carry_cost - rate) * maturity) * norm.cdf(-d1)
        else:
            raise TypeError

    @staticmethod
    def Cash_Or_Nothing(underlying_price, strike_price, cash, maturity, rate, carry_cost, vol, options_type):
        """
        cash or nothing options
        :param underlying_price: asset price
        :param strike_price: strike price
        :param cash: cash given
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: options type
        :return: options price
        """
        d = (np.log(underlying_price / strike_price) + (carry_cost - vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        if options_type.lower() == 'call':
            return cash * np.exp(-rate * maturity) * norm.cdf(d)
        elif options_type.lower() == 'put':
            return cash * np.exp(-rate * maturity) * norm.cdf(-d)
        else:
            raise TypeError

    @staticmethod
    def Two_Assets_Cash_Or_Nothing(
            underlying_price1, underlying_price2, strike_price1, strike_price2,
            cash, maturity, rate, carry_cost1, carry_cost2, vol1, vol2, corr, options_type
    ):
        """
        2 assets cash or nothing options
        :param underlying_price1: asset 1 price
        :param underlying_price2: asset 2 price
        :param strike_pric1: strike price 1
        :param strike_price2: strike price 2
        :param cash: cash amounts
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost1: carry cost 1
        :param carry_cost2: carry cost 2
        :param vol1: volatility on asset 1
        :param vol2: volatitlty on asset 2
        :param corr: correlation between asset 1 and 2
        :param options_type: options type
        inlcluding:
        [1] Cash-or-nothing call
        [2] Cash-or-nothing put
        [3] Cash-or-nothing up-down
        [4] Cash-or-nothing down-up
        :return: options price


        """
        d1 = (np.log(underlying_price1 / strike_price1) + (carry_cost1 - vol1 ** 2 / 2) * maturity) / (
                vol1 * np.sqrt(maturity))
        d2 = (np.log(underlying_price2 / strike_price2) + (carry_cost2 - vol2 ** 2 / 2) * maturity) / (
                vol2 * np.sqrt(maturity))

        if options_type in ['1', 1]:
            return cash * np.exp(-rate * maturity) * CBND(d1, d2, corr)
        elif options_type in ['2', 2]:
            return cash * np.exp(-rate * maturity) * CBND(-d1, -d2, corr)
        elif options_type in ['3', 3]:
            return cash * np.exp(-rate * maturity) * CBND(d1, -d2, -corr)
        elif options_type in ['4', 4]:
            return cash * np.exp(-rate * maturity) * CBND(-d1, d2, -corr)
        else:
            raise TypeError

    @staticmethod
    def Assets_Or_Nothing(underlying_price, strike_price, maturity, rate, carry_cost, vol, options_type):
        """
        assets or nothing options
        :param underlying_price: asset price
        :param strike_price: strike price
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        d = (np.log(underlying_price / strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        if options_type.lower() == 'call':
            return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(d)
        elif options_type.lower() == 'put':
            return underlying_price * np.exp((carry_cost - rate) * maturity) * norm.cdf(-d)
        else:
            raise TypeError

    @staticmethod
    def Super_Share_Options(underlying_price, lower_strike_price, higher_strike_price, maturity, rate, carry_cost, vol):
        """
        super share options model
        :param underlying_price: underlying asset price
        :param lower_strike_price: lower level of strike price
        :param higher_strike_price: higher level of strike price
        :param maturity: time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :return: options price
        """
        d1 = (np.log(underlying_price / lower_strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))
        d2 = (np.log(underlying_price / higher_strike_price) + (carry_cost + vol ** 2 / 2) * maturity) / (
                vol * np.sqrt(maturity))

        return underlying_price * np.exp((carry_cost - rate) * maturity) / lower_strike_price * (
                norm.cdf(d1) - norm.cdf(d2))

    @staticmethod
    def Geometric_Average_Rate_Option(
            underlying_price, avg_price, strike_price, original_maturity, remaining_maturity,
            rate, carry_cost, vol, options_type
    ):
        """
        average price of Asian options
        :param underlying_price: underlying price
        :param avg_price: average price of underlying
        :param strike_price: strike price
        :param original_maturity: original time to maturity
        :param remaining_maturity: remained time to maturity
        :param rate: risk-free rate
        :param carry_cost: carry cost
        :param vol: volatility
        :param options_type: call/put
        :return: options price
        """
        bA = 1 / 2 * (carry_cost - vol ** 2 / 6)
        vA = vol / np.sqrt(3)

        t1 = original_maturity - remaining_maturity

        if t1 > 0:
            strike_price = (
                                       t1 + remaining_maturity) / remaining_maturity * strike_price - t1 / remaining_maturity * avg_price
            return BSM(underlying_price, strike_price, t1, rate, bA, vA, options_type) * remaining_maturity / (
                    t1 + remaining_maturity)
        else:
            return BSM(underlying_price, strike_price, original_maturity, rate, bA, vA, options_type)


if __name__ == '__main__':
    print('execution stock call options', ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15))
    print('execution stock put options',
          ExoticOPtions.Executive_Stock_Options(65, 64, 2, 0.07, 0.04, 0.38, 0.15, 'put'))
    print('forward call options', ExoticOPtions.Forward_Options(60, 1.1, 0.25, 1, 0.08, 0.04, 0.3))
    print('forward put options', ExoticOPtions.Forward_Options(60, 1.1, 0.25, 1, 0.08, 0.04, 0.3, 'put'))
    print('time switch call options',
          ExoticOPtions.maturityime_Switch_Options(100, 110, 5, 1, 0, 0.00274, 0.06, 0.06, 0.26))
    print('time switch put options',
          ExoticOPtions.maturityime_Switch_Options(100, 110, 5, 1, 0, 0.00274, 0.06, 0.06, 0.26, 'put'))
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
    print('exchange options on exchange options 1',
          ExoticOPtions.Exchange_On_Exchange_Options(105, 100, 0.1, 0.75, 1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.5, 1))
    print('exchange options on exchange options 2',
          ExoticOPtions.Exchange_On_Exchange_Options(105, 100, 0.1, 0.75, 1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.5, 2))
    print('exchange options on exchange options 3',
          ExoticOPtions.Exchange_On_Exchange_Options(105, 100, 0.1, 0.75, 1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.5, 3))
    print('exchange options on exchange options 4',
          ExoticOPtions.Exchange_On_Exchange_Options(105, 100, 0.1, 0.75, 1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.5, 4))
    print('max/min exchange options cmin',
          ExoticOPtions.Options_On_Max_Min_Risk_Assets(100, 105, 98, 0.5, 0.05, -0.01, -0.04, 0.11, 0.16, 0.63, 'cmin'))
    print('max/min exchange options cmax',
          ExoticOPtions.Options_On_Max_Min_Risk_Assets(100, 105, 98, 0.5, 0.05, -0.01, -0.04, 0.11, 0.16, 0.63, 'cmax'))
    print('max/min exchange options pmin',
          ExoticOPtions.Options_On_Max_Min_Risk_Assets(100, 105, 98, 0.5, 0.05, -0.01, -0.04, 0.11, 0.16, 0.63, 'pmin'))
    print('max/min exchange options pmax',
          ExoticOPtions.Options_On_Max_Min_Risk_Assets(100, 105, 98, 0.5, 0.05, -0.01, -0.04, 0.11, 0.16, 0.63, 'pmax'))
    print('spread call options', ExoticOPtions.Spread_Options(28, 20, 7, 0.25, 0.05, 0.29, 0.36, 0.42, 'call'))
    print('spread put options', ExoticOPtions.Spread_Options(28, 20, 7, 0.25, 0.05, 0.29, 0.36, 0.42, 'put'))
    print('float lookback call on min options',
          ExoticOPtions.Float_Strike_Lookback_Options(120, 100, 120, 0.5, 0.1, 0.04, 0.3, 'call'))
    print('float lookback put on max options',
          ExoticOPtions.Float_Strike_Lookback_Options(120, 100, 120, 0.5, 0.1, 0.04, 0.3, 'put'))
    print('fixed lookback call on min options',
          ExoticOPtions.Fixed_Strike_Lookback_Options(100, 105, 100, 100, 0.5, 0.1, 0.1, 0.2, 'call'))
    print('fixed lookback put on max options',
          ExoticOPtions.Fixed_Strike_Lookback_Options(100, 105, 100, 100, 0.5, 0.1, 0.1, 0.2, 'put'))
    print('partial float lookback put on call options',
          ExoticOPtions.Partial_Float_LookBack_Options(90, 90, 90, 1, 0.25, 1, 0.06, 0.06, 0.2, 'call'))
    print('partial float lookback put on put options',
          ExoticOPtions.Partial_Float_LookBack_Options(90, 90, 90, 1, 0.25, 1, 0.06, 0.06, 0.2, 'put'))
    print('partial fixed lookback put on call options',
          ExoticOPtions.Partial_Fixed_LookBack_Options(100, 100, 0.5, 1, 0.06, 0.06, 0.1, 'call'))
    print('partial fixed lookback put on put options',
          ExoticOPtions.Partial_Fixed_LookBack_Options(100, 100, 0.5, 1, 0.06, 0.06, 0.1, 'put'))
    print('extreme spread call',
          ExoticOPtions.Extreme_Spread_Options(100, 80, 120, 0.25, 1, 0.1, 0.1, 0.3, 1))
    print('extreme spread put',
          ExoticOPtions.Extreme_Spread_Options(100, 80, 120, 0.25, 1, 0.1, 0.1, 0.3, 2))
    print('reversed extreme spread call',
          ExoticOPtions.Extreme_Spread_Options(100, 80, 120, 0.25, 1, 0.1, 0.1, 0.3, 3))
    print('reversed extreme spread put',
          ExoticOPtions.Extreme_Spread_Options(100, 80, 120, 0.25, 1, 0.1, 0.1, 0.3, 4))
    print('standard barrier options with down-in-call',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'cdi'))
    print('standard barrier options with up-in-call',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'cui'))
    print('standard barrier options with down-in-put',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'pdi'))
    print('standard barrier options with up-in-put',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'pui'))
    print('standard barrier options with down-out-call',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'cdo'))
    print('standard barrier options with up-out-call',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'cuo'))
    print('standard barrier options with down-out-put',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'pdo'))
    print('standard barrier options with up-out-put',
          ExoticOPtions.Standard_Barrier_Options(100, 100, 115, 3, 0.5, 0.08, 0.04, 0.2, 'puo'))
    print('double barrier options with Call up-and-out-down-and-out',
          ExoticOPtions.Double_Barrier_Options(100, 100, 90, 105, 0.25, 0.1, 0.1, 0.25, 'co', 0, 0))
    print('double barrier options with Call Put up-and-out-down-and-out',
          ExoticOPtions.Double_Barrier_Options(100, 100, 90, 105, 0.25, 0.1, 0.1, 0.25, 'po', 0, 0))
    print('double barrier options with Call up-and-in-down-and-in',
          ExoticOPtions.Double_Barrier_Options(100, 100, 90, 105, 0.25, 0.1, 0.1, 0.25, 'ci', 0, 0))
    print('double barrier options with Call Put up-and-in-down-and-in',
          ExoticOPtions.Double_Barrier_Options(100, 100, 90, 105, 0.25, 0.1, 0.1, 0.25, 'pi', 0, 0))
    print('partial barrier options with up-and-out call a',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'cuoa'))
    print('partial barrier options with down-and-out call a ',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'cdoa'))
    print('partial barrier options with up-and-out put a ',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'puoa'))
    print('partial barrier options with down-and-out put a',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'pdoa'))
    print('partial barrier options with down-and-out call carry_cost1',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'cob1'))
    print('partial barrier options with down-and-out put carry_cost1',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'pob1'))
    print('partial barrier options with up-and-out call carry_cost2',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'cuob2'))
    print('partial barrier options with down-and-out call carry_cost2',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'cdob2'))
    print('partial barrier options with up-and-out put carry_cost2',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'puob2'))
    print('partial barrier options with down-and-out put carry_cost2',
          ExoticOPtions.Partial_Barrier_Options(105, 90, 115, 0.35, 0.5, 0.1, 0.05, 0.2, 'pdob2'))
    print('2 assets barrier options with down-and-in call',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'cdi'))
    print('2 assets barrier options with up-and-in call',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'cui'))
    print('2 assets barrier options with down-and-in put',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'pdi'))
    print('2 assets barrier options with up-and-in put',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'pui'))
    print('2 assets barrier options with down-and-out call',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'cdo'))
    print('2 assets barrier options with up-and-out call',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'cuo'))
    print('2 assets barrier options with down-and-out put',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'pdo'))
    print('2 assets barrier options with up-and-out put',
          ExoticOPtions.maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5, 'puo'))

    print('partial 2 assets barrier options with down-and-in call',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'cdi'))
    print('partial 2 assets barrier options with up-and-in call',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'cui'))
    print('partial 2 assets barrier options with down-and-in put',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'pdi'))
    print('partial 2 assets barrier options with up-and-in put',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'pui'))
    print('partial 2 assets barrier options with down-and-out call',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'cdo'))
    print('partial 2 assets barrier options with up-and-out call',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'cuo'))
    print('partial 2 assets barrier options with down-and-out put',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'pdo'))
    print('partial 2 assets barrier options with up-and-out put',
          ExoticOPtions.Partial_maturitywo_Asets_Barrier_Options(100, 100, 95, 90, 0.25, 0.5, 0.08, 0, 0, 0.2, 0.2, 0.5,
                                                                 'puo'))
    print('lookback barrier options with up-and-out call',
          ExoticOPtions.Lookback_Barrier_Options(100, 105, 110, 0.5, 1, 0.1, 0.1, 0.3, 'cuo'))
    print('lookback barrier options with up-and-in call',
          ExoticOPtions.Lookback_Barrier_Options(100, 105, 110, 0.5, 1, 0.1, 0.1, 0.3, 'cui'))
    print('lookback barrier options with down-and-in call',
          ExoticOPtions.Lookback_Barrier_Options(100, 105, 110, 0.5, 1, 0.1, 0.1, 0.3, 'pdi'))
    print('lookback barrier options with down-and-out put',
          ExoticOPtions.Lookback_Barrier_Options(100, 105, 110, 0.5, 1, 0.1, 0.1, 0.3, 'pdo'))
    print('soft barrier options with down-and-in call',
          ExoticOPtions.Soft_Barrier_Options(100, 100, 95, 85, 0.5, 0.1, 0.05, 0.3, 'cdi'))
    print('soft barrier options with down-and-out call',
          ExoticOPtions.Soft_Barrier_Options(100, 100, 95, 85, 0.5, 0.1, 0.05, 0.3, 'cdo'))
    print('soft barrier options with up-and-in put',
          ExoticOPtions.Soft_Barrier_Options(100, 100, 95, 85, 0.5, 0.1, 0.05, 0.3, 'pui'))
    print('soft barrier options with up-and-out put',
          ExoticOPtions.Soft_Barrier_Options(100, 100, 95, 85, 0.5, 0.1, 0.05, 0.3, 'puo'))
    print('gap call options', ExoticOPtions.Gap_Options(50, 50, 57, 0.5, 0.09, 0.09, 0.2, 'call'))
    print('gap put options', ExoticOPtions.Gap_Options(50, 50, 57, 0.5, 0.09, 0.09, 0.2, 'put'))
    print('cash or nothing call options', ExoticOPtions.Cash_Or_Nothing(100, 80, 10, 0.75, 0.06, 0, 0.35, 'call'))
    print('cash or nothing put options', ExoticOPtions.Cash_Or_Nothing(100, 80, 10, 0.75, 0.06, 0, 0.35, 'put'))
    print('2 assets cash or nothing call options',
          ExoticOPtions.Two_Assets_Cash_Or_Nothing(100, 100, 110, 90, 10, 0.5, 0.1, 0.05, 0.06, 0.2, 0.25, 0.5, '1'))
    print('2 assets cash or nothing put options',
          ExoticOPtions.Two_Assets_Cash_Or_Nothing(100, 100, 110, 90, 10, 0.5, 0.1, 0.05, 0.06, 0.2, 0.25, 0.5, '2'))
    print('2 assets cash or nothing up-down options',
          ExoticOPtions.Two_Assets_Cash_Or_Nothing(100, 100, 110, 90, 10, 0.5, 0.1, 0.05, 0.06, 0.2, 0.25, 0.5, '3'))
    print('2 assets cash or nothing down-up options',
          ExoticOPtions.Two_Assets_Cash_Or_Nothing(100, 100, 110, 90, 10, 0.5, 0.1, 0.05, 0.06, 0.2, 0.25, 0.5, '3'))
    print('asset or nothing call options', ExoticOPtions.Assets_Or_Nothing(70, 65, 0.5, 0.07, 0.02, 0.27, 'call'))
    print('asset or nothing put options', ExoticOPtions.Assets_Or_Nothing(70, 65, 0.5, 0.07, 0.02, 0.27, 'put'))
    print('super share options', ExoticOPtions.Super_Share_Options(100, 90, 110, 0.25, 0.1, 0, 0.2))
    print('Asian average price call options',
          ExoticOPtions.Geometric_Average_Rate_Option(80, 80, 85, 0.25, 0.25, 0.05, 0.08, 0.2, 'call'))
    print('Asian average price put options',
          ExoticOPtions.Geometric_Average_Rate_Option(80, 80, 85, 0.25, 0.25, 0.05, 0.08, 0.2, 'put'))