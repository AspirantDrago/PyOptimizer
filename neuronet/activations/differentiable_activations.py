import math


class DifferentiableActivations:
    @staticmethod
    def sigmoida(x, coefficient=1):
        try:
            return 1 / (1 + math.exp(-x * coefficient))
        except:
            if x > 0:
                return 1.0
            else:
                return 0.0

    @staticmethod
    def sigmoida_diff(x, coefficient=1):
        try:
            return DifferentiableActivations.sigmoida(x, coefficient) * (1 - DifferentiableActivations.sigmoida(x, coefficient))
        except:
            return 0

    @staticmethod
    def tang(x, coefficient=1):
        try:
            return math.tanh(x * coefficient)
        except:
            if x > 0:
                return 1.0
            else:
                return -1.0

    @staticmethod
    def tang_diff(x, coefficient=1):
        try:
            return - coefficient * (math.tanh(coefficient * x) ** 2 - 1)
        except:
            return 0

    @staticmethod
    def arctang(x, coefficient=1):
        try:
            return math.atan(x * coefficient)
        except:
            if x > 0:
                return 1.0
            else:
                return -1.0

    @staticmethod
    def arctang_diff(x, coefficient=1):
        try:
            return coefficient / ((coefficient * x) ** 2 + 1)
        except:
            return 0
