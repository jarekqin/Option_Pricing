from options_pricing_model_class.abstract_basic import BasicFiniteDifferences
from options_pricing_model_class.finite_difference_basic_class import FiniteDifferenceBasicClass
import numpy as np


class FDExplicitEU(BasicFiniteDifferences,FiniteDifferenceBasicClass):
    def __init__(self,s0,k,r,t,sigma,smax,m,n,is_put=False):
        super().__init__(s0,k,r,t,sigma,smax,m,n,is_put)

    def setup_bounday_conditions(self):
        is s