import math
import random
from optimizer import Optimizer
from optimizer import _Distribs, _WidthVariation


class OptimizerDE(Optimizer, _Distribs, _WidthVariation):
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
                 const_f=1,
                 const_d=0.5,
                 distrib=None,
                 maximization=False,
                 limits_in=[],
                 limits_out=[],
                 width_variation=[],
                 single_width_variation=1
                 ):
        super(OptimizerDE, self).__init__()
        self._const_f = const_f
        self._const_d = const_d
        self._distrib = distrib
        if self._distrib is None:
            self._distrib = self.distrib_normal
        self._width_variation = list(width_variation)
        self._single_width_variation = float(single_width_variation)

    def generator(self):
        """Генерирует случайный вектор входных значений

        :return: Список значений входного вектора (list)
        """
        while True:
            lst = []
            for i in range(self.count):
                if self.var[i]:
                    val = self._distrib(self._width_variation[i], self)
                    center = (self.intervals[i][1] + self.intervals[i][0]) / 2
                    val = center + val * (self.intervals[i][1] - self.intervals[i][0]) / 2
                    val = max(self.intervals[i][0], min(self.intervals[i][1], val))
                    lst.append(val)
            if all(lim(lst) for lim in self.limits_in):
                return lst

    def new_iter(self):
        new_gens = {}
        gen_count = len(self.generation)
        for i in range(gen_count):
            gen = self.generation[i]
            arr = random.sample(list(filter(lambda x: x != i, range(gen_count))), 3)
            arr = [self.generation[x] for x in arr]
            mutable = gen[:]
            for j in range(len(mutable)):
                if random.random() >= self._const_d:
                    mutable[j] = arr[0][j] + self._const_f * (arr[1][j] - arr[2][j])
                mutable[j] = max(self.intervals[j][0], min(self.intervals[j][1], mutable[j]))
            e1 = self._func(gen, self._model(gen))
            e2 = self._func(mutable, self._model(mutable))
            if e2 < e1 and all(lim(mutable) for lim in self.limits_in):
                new_gens[i] = mutable[:]
        for key in new_gens.keys():
            self.generation[key] = new_gens[key]
            out2 = self._model(self.generation[key])
            e2 = self._func(self.generation[key], out2)
            all_limits_out = all(lim(out2) for lim in self.limits_out)
            if (e2 < self.optimal_err) and all_limits_out:
                if not self._tol or ((self.optimal_err - self.energy) >= self._tol_limit):
                    self._idle_step = 0
                self.val = self.generation[key]
                self.optimal_err = self.error
                self.optimal_y = out2
                self.optimal = self.val.copy()
        if self._print_log:
            print(f'step: {self._step}\terror: {self.error}\ninput: {self.val}\noutput: {self.out}\n')
        super(OptimizerDE, self).new_iter()
        return self

    def initialize(self, arr, reset_intervals=True):
        """Инициализирует оптимизатор начальным вектором входных значений

        :param val: вектор входных значений (list)
        """
        self.generation = [list(x) for x in arr]
        self.initial = self.generation[0]
        self.count = len(self.initial)
        self.var = [False] * self.count
        if reset_intervals:
            self.intervals = [[None, None] for _ in range(self.count)]
        self.reset(reset_trands=False)
        return self

    def initialize_random(self, count):
        arr = [self.generator() for i in range(count)]
        print(arr, self.count)
        return self.initialize(arr, reset_intervals=False)

    def save(self):
        """Сохраняет текущее состояние как начальное"""
        return self
