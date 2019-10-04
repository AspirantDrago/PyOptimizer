from neuronet import ActivePerceptron
import numpy as np


class NumPyPerceptron(ActivePerceptron):
    def _gety(self, x):
        prev_layer = x
        for i in range(self.count_layers):
            prev_layer_np = np.array([1] + prev_layer)
            row = np.dot(self.layers[i], prev_layer_np)
            prev_layer = list(map(self.activate, row))
        return prev_layer

    def _gety_numpy(self, prev_layer):
        for i in range(self.count_layers):
            prev_layer = np.insert(prev_layer, 0, 1)
            prev_layer = np.dot(self.layers[i], prev_layer)
            prev_layer = np.vectorize(self.activate)(prev_layer)
        return prev_layer

    def gety_numpy(self, x, is_norm=True):
        self._token += 1
        prev_layer = x.copy()
        if is_norm:
            prev_layer = np.array([self.normalize(prev_layer[i], self.inp_norm_from[i], self.inp_norm_to[i])
                          for i in range(len(prev_layer))])
        out = self._gety_numpy(prev_layer)
        if is_norm:
            out = np.array([self.normalize(out[i], self.out_norm_from[i], self.out_norm_to[i]) for i in
                   range(len(out))])
        return out

    def setlayers(self, layers):
        self.layers = np.array(layers)
        self.count_layers = len(layers)
        return self

    def _getlayers(self):
        return self.layers.tolist()
