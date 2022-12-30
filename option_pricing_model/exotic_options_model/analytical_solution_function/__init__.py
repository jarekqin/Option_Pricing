
from option_pricing_model.options_tools.tools import *
from option_pricing_model.vanilla_options_model.bsm_base import BSM


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
    def Two_Asset_Corr_Options(underlying_price1, underlying_price2, strike_price1, strike_price2, maturity,
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
        :param carry_cosmaturity1: carry cost
        :param carry_cost2: carry cost
        :param vol1: volatitlity 1
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
                        -d1 + 2 * carry_cost * np.sqrt(maturity) / vol - g1, np.sqrt(lookback_length / maturity)) - \
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
                            2 * carry_cost / vol ** 2) * \
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
          ExoticOPtions.Partial_Fixed_LookBack_Options(100, 100, 0.5, 1, 0.06,0.06, 0.1, 'call'))
    print('partial fixed lookback put on put options',
          ExoticOPtions.Partial_Fixed_LookBack_Options(100, 100, 0.5, 1, 0.06,0.06, 0.1, 'put'))

