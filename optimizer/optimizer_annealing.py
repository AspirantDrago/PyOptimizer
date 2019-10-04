import math
import random
from optimizer import Optimizer
from optimizer import _Distribs, _WidthVariation


class OptimizerAnnealing(Optimizer, _Distribs, _WidthVariation):
    """Класс для создания оптимизаторов по методу имитации отжига"""

    def __init__(self,
                 model=None,
                 func=None,
                 print_log=False,
                 epoches=True,
                 epoches_limit=10000,
                 idle=False,
                 idle_limit=1000,
                 cond_limit=None,
                 tol=False,
                 tol_limit=0.001,
                 t0=1,
                 method=None,
                 distrib=None,
                 maximization=False,
                 limits_in=[],
                 limits_out=[],
                 width_variation=[],
                 single_width_variation=1
                 ):
        """Конструктор класса, создающий оптимизатор по методу имитации отжига

        :param model: математическая модель
            Функция, преобразующая вектор входных значений в вектор выходных значений:
            model(x)
                :param x: вектор входных значений (list)
                :return: вектор выходных значений (list)
        :type model: function

        :param func: функционал оптимизации
            Функция, ставящая в соответствие входному и выходному векторам
            определенное значение (чаще всего - ошибки):
            func(x, y)
                :param x: вектор входных значений (list)
                :param y: вектор выходных значений (list)
                :return: значение функционала (float)
        :type func: function

        :param print_log: Печать логов в стандартный поток. По умолчанию - False
        :type print_log: bool

        :param epoches: Ограничение количества эпох. По умолчанию - True
        :type epoches: bool

        :param epoches_limit: Максимальное количество эпох. По умолчанию 10000
        :type epoches_limit: int

        :param idle: Ограничение количества холостых эпох. По умолчанию - False
            Холостая эпоха - когда не было найдено более оптимальное состояние,
            по сравнению с предыдущими эпохами
        :type idle: bool

        :param idle_limit: Максимальное количество холостых эпох. По умолчанию 1000
        :type idle_limit: int

        :param cond_limit: Функция условия останова процесса оптимизации
            cond_limit(x, y, e)
                :param x: вектор входных значений (list)
                :param y: вектор выходных значений (list)
                :param e: значение функционала оптимизации (float)
                :return: следует ли прерывать процесс оптимизации (bool)
        :type cond_limit: function

        :param tol: Ограничение по точности сходимости. По умолчанию - False
            Если в процессе оптимизации произошло изменение менее, чем на значение точности,
            то данная эпоха всё равно будет считаться холостой.
        :type tol: bool

        :param tol_limit: Ограничение по точности сходимости. По умолчанию - 0.001
            Если в процессе оптимизации произошло изменение менее, чем на значение точности,
            то данная эпоха всё равно будет считаться холостой.
        :type tol_limit: float

        :param t0: Начальная температура имитации отжига. По умолчанию 1
        :type t0: float

        :param method: Функция пересчёта температуры. По умолчанию - method_boltzman
            method(t0, step, obj)
                :param t0: начальная температура имитации отжига (float)
                :param step: номер текущей итерации (int)
                :param obj: ссылка на текущий объект оптимизатора (optimizer_annealing). Опционально
                :type: float
            Доступны следующие стандартные значения:
                method_boltzman - рас
                method_koshi -
        :type method: function

        :param distrib: Функция распределения случайной величины. По умолчанию - distrib_normal
            distrib(width, obj)
                :param width: начальная температура имитации отжига (float)
                :param obj: ссылка на текущий объект оптимизатора (optimizer_annealing). Опционально
                :type: float
        :type distrib: function




        :param maximization:
        :param limits_in:
        :param limits_out:
        width_variation
        single_width_variation
        """
        super(OptimizerAnnealing, self).__init__()
        self._t0 = float(t0)
        self._method = method
        if self._method is None:
            self._method = self.method_boltzman
        self._distrib = distrib
        if self._distrib is None:
            self._distrib = self.distrib_normal
        self._width_variation = list(width_variation)
        self._single_width_variation = float(single_width_variation)
        self.trands = {'x': [], 'y': [], 'e': [], 'opt_x': [], 'opt_y': [], 'opt_e': [], 't': []}

    def initialize(self, val):
        if not self._width_variation:
            self.full_width_variation()
        return super(OptimizerAnnealing, self).initialize(val)

    def generator(self):
        """Генерирует случайный вектор входных значений

        :return: Список значений входного вектора (list)
        """
        while True:
            lst = self.val.copy()
            for i in range(self.count):
                if self.var[i]:
                    if self.t0 == 0:
                        val = 0.0
                    else:
                        val = self._distrib(self._width_variation[i], self) * self.temperature / self.t0
                    val = self.val[i] + val * (self.intervals[i][1] - self.intervals[i][0]) / 2
                    val = max(self.intervals[i][0], min(self.intervals[i][1], val))
                    lst[i] = val
            if all(lim(lst) for lim in self.limits_in):
                return lst

    @staticmethod
    def method_koshi(t0, step, obj=None):
        return t0 / (1 + step)

    @staticmethod
    def method_boltzman(t0, step, obj=None):
        return t0 / (1 + math.log(step, math.e))

    def new_iter(self):
        """Выполняет 1 итерации по методу имитации отжига"""
        new = self.generator()
        out2 = self._model(new)
        e2 = self._func(new, out2)
        dE = e2 - self.energy
        if self.maximization:
            dE *= -1
        if self._print_log:
            print(f'step: {self._step}\terror: {self.energy}\tt: {self.temperature}\ninput: {new}\noutput: {out2}\n')
        all_limits_out = all(lim(out2) for lim in self.limits_out)
        if (dE < 0 and all_limits_out) or (random.random() < self.probability(dE)):
            self.val = new
            if (self.energy < self.optimal_err) and all_limits_out:
                if not self._tol or ((self.optimal_err - self.energy) >= self._tol_limit):
                    self._idle_step = 0
                self.optimal_err = self.energy
                self.optimal_y = out2
                self.optimal = self.val.copy()
        self.trands['t'].append(self.temperature)
        super(OptimizerAnnealing, self).new_iter()
        return self

    def probability(self, dE):
        """Вероятность перехода системы в новое состояние

        :param dE: изменение энергии системы при переходе (float)
        :return: веростность перехода в интервале [0.0, 1.0] (float)
        """
        if dE <= 0:
            return 1.0
        if self.temperature == 0:
            return 0.0
        try:
            return 1 / (1 + math.exp(dE / self.temperature))
        except:
            return 0.0

    def _get_distrib(self):
        """"""
        return self._distrib

    def _set_distrib(self, value):
        self._distrib = value

    distrib = property(_get_distrib, _set_distrib)

    @property
    def energy(self):
        """Теущее значение энергии системы (float). Только для чтения
        То же самое, что и текущее начение функционала ошибки
        """
        return self._func(self.val, self.out)

    @property
    def error(self):
        """Теущее значение функционала ошибки (float). Только для чтения"""
        return self.energy

    def _get_method(self):
        """"""
        return self._method

    def _set_method(self, value):
        self._method = value

    method = property(_get_method, _set_method)

    @property
    def temperature(self):
        """Текущая температура системы (float, >= 0). Только для чтения"""
        return self._method(self._t0, self._step, self)

    def _get_t0(self):
        """Начальная температура имитации отжига
        :rtype: float
        """
        return self._t0

    def _set_t0(self, value):
        self._t0 = value

    t0 = property(_get_t0, _set_t0)
    # TODO Дописать single_width_variation
