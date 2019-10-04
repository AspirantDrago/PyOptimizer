import random


class _Distribs:
    @staticmethod
    def distrib_normal(width, obj=None):
        return random.normalvariate(0, width)

    @staticmethod
    def distrib_uniform(width, obj=None):
        return (2 * random.random() - 1) * width
