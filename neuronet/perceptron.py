from neuronet import ActivePerceptron, Neuron
from neuronet.teachers.back_propagation import BackPropagation


class Perceptron(ActivePerceptron, BackPropagation):
    def __init__(self, count_x=0, count_y=0, layers=None):
        super().__init__(count_x=count_x, count_y=count_y)
        if layers is None:
            layers = []
        self.setlayers(layers + [count_y], empty=True)

    def _gety(self, x):
        for i in range(len(x)):
            self.input_layer[i].value = x[i]
        return [n.getvalue(self._token) for n in self.output_layer]

    def setlayers(self, layers, empty=False):
        self.layers = []
        self.layers.append([Neuron(position=(0, _)) for _ in range(self.count_x)])
        for layer in layers:
            if empty:
                row = [Neuron(position=(len(self.layers), i), inputs=self.output_layer, activation=self.activate) for i in range(layer)]
            else:
                row = [Neuron(position=(len(self.layers), i), weights=layer[i], inputs=self.output_layer, activation=self.activate) for i in range(len(layer))]
            self.layers.append(row)
        self.count_layers = len(self.layers)
        return self

    def _prev_learn(self):
        return self

    def _post_learn(self):
        for layer in self.layers:
            for n in layer:
                n.update()
        return self

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    def _getlayers(self):
        return [[neuron._weights for neuron in layer ] for layer in self.layers[1:]]
