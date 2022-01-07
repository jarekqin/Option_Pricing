import math
from abc import ABC,abstractmethod

class BasicTreeOptionsClass(ABC):
    @abstractmethod
    def setup_parameters(self):
        pass

    @abstractmethod
    def price(self):
        pass

    @abstractmethod
    def init_stock_price_tree(self):
        pass

    @abstractmethod
    def init_payoffs_tree(self):
        pass

    @abstractmethod
    def traverse_tree(self,payoffs):
        pass


class FiniteDifferencesClass(ABC):
    @abstractmethod
    def setup_boundary_conditions(self):
        pass

    @abstractmethod
    def traverse_grid(self):
        pass

    @abstractmethod
    def interpolate(self):
        pass

