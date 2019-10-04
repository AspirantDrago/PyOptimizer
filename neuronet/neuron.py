from neuronet.activations import Activations
import random


class Neuron:
    def __init__(self,
                 position=tuple(),
                 weights=None,
                 count=None,
                 activation=None,
                 inputs=None,
                 fixvalue=None,
                 coefficient=1,
                 speed_learn=0.1,
                 inertia=0.9,
                 initial_amplitude=1):
        if count is None:
            if inputs:
                self._count = len(inputs)
            elif weights:
                self._count = len(weights)
            else:
                self._count = 0
        else:
            self._count = count
        if weights is None:
            weights = [random.uniform(-initial_amplitude, initial_amplitude) for _ in range(self._count + 1)]
        self._weights = weights
        if inputs is None:
            inputs = [None] * self._count
        self._inputs = [None] + inputs
        if activation is None:
            activation = Activations.sigmoida
        self.activation = activation
        self._fixvalue = fixvalue
        self._token = None
        self._token_error = None
        self._last_value = 0
        self._coefficient = coefficient
        self._delta_weights = [0] * (self._count + 1)
        self._delta_weights_prev = [0] * (self._count + 1)
        self._speed_learn = speed_learn
        self._inertia = inertia
        self._error = 0
        self._position = position
        self._initial_amplitude = initial_amplitude

    def update(self):
        for i in range(self._count + 1):
            self._weights[i] += self._speed_learn * (self._delta_weights[i] + self._inertia * self._delta_weights_prev[i])
        self._delta_weights_prev = self._delta_weights.copy()
        self._delta_weights = [0] * (self._count + 1)
        self._error = None

    def __getitem__(self, item):
        return self._weights[item]

    def __setitem__(self, key, value):
        self._weights[key] = value

    def adder(self):
        result = self[0]
        for i in range(1, self._count + 1):
            if self._inputs[i]:
                result += self._inputs[i].getvalue(self._token) * self[i]
        return result

    def activator(self):
        return self._activation(self.adder(), self._coefficient)

    def diff_activator(self):
        return self._diff_activation(self.adder(), self._coefficient)

    def getvalue(self, token=None):
        if not (self._fixvalue is None):
            return self._fixvalue
        if not (token and token == self._token):
            self._token = token
            self._last_value = self.activator()
        return self._last_value

    def _get_value(self):
        return self.getvalue()

    def _set_value(self, val):
        self._fixvalue = val

    value = property(_get_value, _set_value)

    def _get_coefficient(self):
        return self._coefficient

    def _set_coefficient(self, value):
        self._coefficient = value

    coefficient = property(_get_coefficient, _set_coefficient)

    def _set_count(self, count):
        def align_list(lst, count, amplitude=0):
            del lst[count + 1:]
            if amplitude:
                lst.extend([random.uniform(-amplitude, amplitude) for _ in range(count + 1 - len(lst))])
            else:
                lst.extend([0] * (count + 1 - len(lst)))

        self._count = count
        align_list(self._weights, self._count, self._initial_amplitude)
        align_list(self._inputs, self._count)
        align_list(self._delta_weights, self._count)
        align_list(self._delta_weights_prev, self._count)

    def _get_count(self):
        return self._count

    count = property(_get_count, _set_count)

    def _get_inertia(self):
        return self._inertia

    def _set_inertia(self, value):
        self._inertia = value

    inertia = property(_get_inertia, _set_inertia)

    def _get_speed_learn(self):
        return self._speed_learn

    def _set_speed_learn(self, value):
        self._speed_learn = value

    speed_learn = property(_get_speed_learn, _set_speed_learn)

    def _get_weights(self):
        return self._weights

    def _set_weights(self, values):
        self._weights = list(values)
        self.count = len(self._weights)

    weights = property(_get_weights, _set_weights)

    def _set_inputs(self, values):
        self._inputs = [None] + list(values)
        self._count = len(self._inputs) - 1

    def _get_inputs(self):
        return self._inputs

    inputs = property(_get_inputs, _set_inputs)

    def _get_activation(self):
        return self._activation

    def _set_activation(self, func):
        self._activation = func
        diff_name = self._activation.__name__ + '_diff'
        self._diff_activation = lambda x, coefficient: 0
        if hasattr(Activations, diff_name):
            self._diff_activation = getattr(Activations, diff_name)

    activation = property(_get_activation, _set_activation)

    def __repr__(self):
        return f'Neuron {self._position}'
        # return f'Neuron {self._position}: size={self._count}, value={self.value} {self._last_value} {self._token}'

    def _get_error(self):
        return self._error

    def _set_error(self, value):
        if not (self._fixvalue) and (self._error is None):
            self._error = value * self.diff_activator()
            self._delta_weights[0] += self._error
            for i in range(1, 1 + self._count):
                self._delta_weights[i] += self._error * self._inputs[i].getvalue(self._token)
            for i in range(1, 1 + self._count):
                if self._inputs[i]:
                    self._inputs[i].error = self._error * self._weights[i]

    error = property(_get_error, _set_error)
