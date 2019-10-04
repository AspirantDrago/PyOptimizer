import math
import random


class Optimizer:
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
                 ):
        """Конструктор класса, создающий оптимизатор

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

        :param maximization:
        :param limits_in:
        :param limits_out:
        """
        self._model = model
        self._func = func
        self._print_log = bool(print_log)
        self._epoches = bool(epoches)
        self._epoches_limit = int(epoches_limit)
        self._idle = bool(idle)
        self._idle_limit = int(idle_limit)
        self._cond_limit = cond_limit
        self._tol = bool(tol)
        self._tol_limit = float(tol_limit)
        self._step = 1
        self.count = 0
        self.intervals = []
        self.var = []
        self.maximization = bool(maximization)
        self.optimal = []
        self.optimal_y = []
        self.optimal_err = float('inf')
        self.limits_in = list(limits_in)
        self._idle_step = 0
        self.limits_out = list(limits_out)
        self.trands = {'x': [], 'y': [], 'e': [], 'opt_x': [], 'opt_y': [], 'opt_e': []}

    def initialize(self, val):
        """Инициализирует оптимизатор начальным вектором входных значений

        :param val: вектор входных значений (list)
        """
        self.initial = list(val)
        self.count = len(val)
        self.var = [False] * self.count
        self.intervals = [[None, None] for _ in range(self.count)]
        self.reset(reset_trands=False)
        return self

    def generator(self):
        """Генерирует случайный вектор входных значений

        :return: Список значений входного вектора (list)
        """
        return self.val[:]

    def new_iter(self):
        self.trands['x'].append(self.val)
        self.trands['y'].append(self.out)
        self.trands['e'].append(self.error)
        self.trands['opt_x'].append(self.optimal)
        self.trands['opt_y'].append(self.optimal_y)
        self.trands['opt_e'].append(self.optimal_err)
        self._step += 1
        self._idle_step += 1
        return self

    def reset(self, reset_trands=False):
        """Сбрасывает оптимизатор на начальное состояние

        :param reset_trands: Очищает тренды (True/False). По умолчанию - False
        """
        self._step = 1
        self._idle_step = 0
        self.val = self.initial
        self.optimal = self.initial
        self.optimal_y = self.out
        self.optimal_err = self.error
        if reset_trands:
            for key in self.trands:
                self.trands[key] = []
        return self

    def save(self):
        """Сохраняет текущее состояние как начальное"""
        self.initialize(self.optimal.copy())
        return self

    def start(self, no_reset=False):
        """Запускает процесс оптимизации по методу имитации отжига

        :param no_reset: Не сбрасывать оптимизатор к начальному состоянию (True/False).
            По умолчанию - False
        :return: None
        """
        if not no_reset:
            self.reset()
        while not (
                (self._epoches and (self._step > self._epoches_limit))
                or (self._idle and (self._idle_step > self._idle_limit))
                or (not(self.cond_limit is None) and self.cond_limit(self.val, self.out, self.error))
        ):
            self.new_iter()
        return self

    def _get_cond_limit(self):
        """Функция условия останова процесса оптимизации (function)
        cond_limit(x, y, e)
            :param x: вектор входных значений (list)
            :param y: вектор выходных значений (list)
            :param e: значение функционала оптимизации (float)
            :return: следует ли прерывать процесс оптимизации (bool)
        """
        return self._cond_limit

    def _set_cond_limit(self, value):
        self._cond_limit = value

    cond_limit = property(_get_cond_limit, _set_cond_limit)

    def _get_epoches(self):
        """Ограничение количества эпох (True/False)"""
        return self._epoches

    def _set_epoches(self, value):
        self._epoches = value

    epoches = property(_get_epoches, _set_epoches)

    def _get_epoches_limit(self):
        """Максимальное количество эпох (int)"""
        return self._epoches_limit

    def _set_epoches_limit(self, value):
        self._epoches_limit = value

    epoches_limit = property(_get_epoches_limit, _set_epoches_limit)

    @property
    def error(self):
        """Теущее значение функционала ошибки (float). Только для чтения"""
        return self._func(self.val, self.out)

    def _get_func(self):
        """функционал оптимизации (function)
        Функция, ставящая в соответствие входному и выходному векторам
        определенное значение (чаще всего - ошибки):
        func(x, y)
            :param x: вектор входных значений (list)
            :param y: вектор выходных значений (list)
            :return: значение функционала (float)
        """
        return self._func

    def _set_func(self, value):
        self._func = value

    func = property(_get_func, _set_func)

    def _get_idle(self):
        """Ограничение количества холостых эпох (True/False)
        Холостая эпоха - когда не было найдено более оптимальное состояние,
        по сравнению с предыдущими эпохами
        """
        return self._idle

    def _set_idle(self, value):
        self._idle = value

    idle = property(_get_idle, _set_idle)

    def _get_idle_limit(self):
        """Максимальное количество холостых эпох (int)
        Холостая эпоха - когда не было найдено более оптимальное состояние,
        по сравнению с предыдущими эпохами
        """
        return self._idle_limit

    def _set_idle_limit(self, value):
        self._idle_limit = value

    idle_limit = property(_get_idle_limit, _set_idle_limit)

    def _get_model(self):
        """Математическая модель (function)
        Функция, преобразующая вектор входных значений в вектор выходных значений:
        model(x)
            :param x: вектор входных значений (list)
            :return: вектор выходных значений (list)
        """
        return self._model

    def _set_model(self, value):
        self._model = value

    model = property(_get_model, _set_model)

    @property
    def out(self):
        """Теущее значение вектора выходных данных (list). Только для чтения"""
        return self._model(self.val)

    def _get_print_log(self):
        """Печать логов в стандартный поток (True/False)"""
        return self._print_log

    def _set_print_log(self, value):
        self._print_log = value

    print_log = property(_get_print_log, _set_print_log)

    def _get_tol(self):
        """Ограничение по точности сходимости (True/False).
        Если в процессе оптимизации произошло изменение менее, чем на значение точности,
        то данная эпоха всё равно будет считаться холостой.
        """
        return self._tol

    def _set_tol(self, value):
        self._tol = value

    tol = property(_get_tol, _set_tol)

    def _get_tol_limit(self):
        """Ограничение по точности сходимости (float).
        Если в процессе оптимизации произошло изменение менее, чем на значение точности,
        то данная эпоха всё равно будет считаться холостой.
        """
        return self._tol_limit

    def _set_tol_limit(self, value):
        self._tol_limit = value

    tol_limit = property(_get_tol_limit, _set_tol_limit)
