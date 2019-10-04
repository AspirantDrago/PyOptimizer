from neuronet import Perceptron, Neuron
from neuronet.teachers.back_propagation import BackPropagation
import random
import hashlib


class PseudoRandomNeuroNet(Perceptron, BackPropagation):
    def __init__(self,
                 count_x=0,
                 count_y=0,
                 count_hidden=0,
                 mass=0,
                 key=None,
                 is_duplicate=False,
                 is_loop=False):
        super().__init__(count_x=count_x, count_y=count_y)
        self._count_hidden = count_hidden
        self._mass = mass
        self._min_key = 1
        self._max_key = 10 ** 9
        if key is None:
            key = random.randint(self._min_key, self._max_key)
        self._key = key
        self._is_duplicate = is_duplicate
        self._is_loop = is_loop
        self.update()

    def gethash(self, value, iter=1):
        return int(hashlib.md5(str(value * iter).encode('utf-8')).hexdigest(), 16)

    def update(self):
        self._size = self.count_x + self.count_y + self._count_hidden
        self.layers = [[Neuron(position=(0, i), count=self._mass) for i in range(self._size)]]
        for j in range(self.count_x):
            self.layers[0][j].value = 0
        for j in range(self.count_x, self._size):
            arr = [None] * self._mass
            for i in range(self._mass):
                x = self._key + i + j * self._mass
                iter = 1
                while True:
                    k = self.gethash(x, iter) % self._size
                    if (self._is_duplicate or not (k in arr)) and (self._is_loop or k != j):
                        break
                    iter += 1
                arr[i] = k
            self.layers[0][j].inputs = [self.layers[0][i] for i in arr]






    def _gety(self, x):
        for i in range(self.count_x):
            self.layers[0][i].value = x[i]
        return [n.getvalue(self._token) for n in self.layers[0][self.count_x: self.count_x + self.count_y]]

    @property
    def input_layer(self):
        return self.layers[0][:self.count_x]

    @property
    def output_layer(self):
        return self.layers[0][self.count_x: self.count_x + self.count_y]
