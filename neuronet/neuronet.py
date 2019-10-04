from neuronet.activations import Activations
import json


class NeuroNet(Activations):
    def __init__(self, count_x=0, count_y=0):
        self.krutizna = 1
        self.activate = NeuroNet.sigmoida
        self.setsize(count_x, count_y)
        self._token = 0
        self._error_power = 2
        self.file_name = None

    def setfilename(self, filename):
        self.file_name = filename

    def savetofile(self, filename=None):
        if filename is None:
            filename = self.file_name
        f = open(filename, "w", encoding='utf-8')
        print(self.neurodata)
        f.write(json.dumps(self.neurodata))
        f.close()

    def loadfromfile(self, filename):
        self.file_name = filename
        # TODO

    def setsize(self, count_x, count_y):
        self.count_x = count_x
        self.count_y = count_y
        self.inp_norm_from = [[-1, 1] for i in range(count_x)]
        self.inp_norm_to = [[-1, 1] for i in range(count_x)]
        self.out_norm_from = [[0, 1] for i in range(count_y)]
        self.out_norm_to = [[0, 1] for i in range(count_y)]
        return self

    def _gety(self, x):
        return x[:]

    def gety(self, x, is_norm=True):
        self._token += 1
        prev_layer = list(x)
        if is_norm:
            prev_layer = [self.normalize(prev_layer[i], self.inp_norm_from[i], self.inp_norm_to[i]) for i in range(len(prev_layer))]
        out = self._gety(prev_layer)
        if is_norm:
            out = [self.normalize(out[i], self.out_norm_from[i], self.out_norm_to[i]) for i in range(len(out))]
        return out

    def error(self, x, y, is_norm=True):
        y_out = self.gety(x=x, is_norm=is_norm)
        if is_norm:
            y_norm = [self.normalize(y[i], self.out_norm_to[i], self.out_norm_from[i]) for i in range(len(y))]
            y_out_norm = [self.normalize(y_out[i], self.out_norm_to[i], self.out_norm_from[i]) for i in range(len(y))]
        else:
            y_norm = y[:]
            y_out_norm = y_out[:]
        err = 0
        for i in range(min(len(y_norm), len(y_out_norm))):
            err += abs(y_norm[i] - y_out_norm[i]) ** self._error_power
        err /= self._error_power
        return err, y_out

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]
        return self.gety(args, kwargs.get('is_norm', True))

    def normalize(self, value, in_interval, out_interval):
        return out_interval[0] + (value - in_interval[0]) * (out_interval[1] - out_interval[0]) /\
               (in_interval[1] - in_interval[0])

    def setinpnormfrom(self, inp_norm_from):
        self.inp_norm_from = inp_norm_from
        return self

    def setinpnormto(self, inp_norm_to):
        self.inp_norm_to = inp_norm_to
        return self

    def setoutnormfrom(self, out_norm_from):
        self.out_norm_from = out_norm_from
        return self

    def setoutnormto(self, out_norm_to):
        self.out_norm_to = out_norm_to
        return self

    def loader(self, dc):
        if 'count_x' in dc:
            self.count_x = dc['count_x']
        if 'count_y' in dc:
            self.count_y = dc['count_y']
        if 'count_x' in dc and 'count_y' in dc:
            self.setsize(self.count_x, self.count_y)
        if 'krutizna' in dc:
            self.krutizna = dc['krutizna']
        if 'activate' in dc:
            if dc['activate'] == 'afnHyperTan':
                self.activate = NeuroNet.tang
            if dc['activate'] == 'afnArcTan':
                self.activate = NeuroNet.arctang
            else:
                self.activate = NeuroNet.sigmoida
        if 'inp_norm_from' in dc:
            self.setinpnormfrom(dc['inp_norm_from'])
        if 'inp_norm_to' in dc:
            self.setinpnormto(dc['inp_norm_to'])
        if 'out_norm_from' in dc:
            self.setoutnormfrom(dc['out_norm_from'])
        if 'out_norm_to' in dc:
            self.setoutnormto(dc['out_norm_to'])
        return self

    def _getlayers(self):
        return [[[]]]

    @property
    def neurodata(self):
        layers = self._getlayers()
        return {
            'layers': layers,
            'layers_count': len(layers),
            'count_x': self.count_x,
            'count_y': self.count_y,
            'krutizna': self.krutizna,
            'activate': self.activate,
            'inp_norm_from': self.inp_norm_from,
            'inp_norm_to': self.inp_norm_to,
            'out_norm_from': self.out_norm_from,
            'out_norm_to': self.out_norm_to,
        }
