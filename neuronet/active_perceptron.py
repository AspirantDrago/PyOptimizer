from neuronet import NeuroNet


class ActivePerceptron(NeuroNet):
    def __init__(self, count_x=0, count_y=0):
        super(ActivePerceptron, self).__init__(count_x=count_x, count_y=count_y)
        self.layers = []
        self.count_layers = 0

    def _gety(self, x):
        prev_layer = [1] + x
        for i in range(self.count_layers):
            row = [sum(layer[j] * prev_layer[j] for j in range(len(prev_layer))) for layer in self.layers[i]]
            prev_layer = [1] + list(map(self.activate, row))
        return prev_layer[1:]

    def setlayers(self, layers):
        self.layers = layers
        self.count_layers = len(layers)
        return self

    def loader(self, dc):
        super(ActivePerceptron, self).loader(dc)
        if 'layers_count' in dc:
            self.count_layers = dc['layers_count']
        if 'layers' in dc:
            self.setlayers(dc['layers'])
        return self

    def _getlayers(self):
        return self.layers
