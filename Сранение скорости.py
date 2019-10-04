from neuronet import ActivePerceptron, Perceptron, NumPyPerceptron
from deductor_parser import DeductorParser
from timeit import Timer
import numpy as np


x = [3, 160, 0.5, 38.5, 77.7, 302.5, 700, 8]
np_x = np.array(x)

parser = DeductorParser('C:\\Users\\Aleksandr\\Desktop\\Электролиз\\Данные\\дед маш.ded')
parser.setdocument('Текстовый файл (C:\\Users\\Николай\\Downloads\\DataSet (2).csv)')
parser.setneuronet('Нейросеть [8 x75 х75  x 4]', parse=True)

activ_nn = ActivePerceptron().loader(parser.neurodata)

print(*activ_nn(x))

nn = Perceptron().loader(parser.neurodata)
print(*nn(x))

numpy_nn = NumPyPerceptron().loader(parser.neurodata)
print(*numpy_nn(x))
print(numpy_nn.gety_numpy(np_x))

t = Timer(lambda: activ_nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: numpy_nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: numpy_nn.gety_numpy(np_x))
print (t.timeit(number=1000))
print()


t = Timer(lambda: activ_nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: numpy_nn(x))
print (t.timeit(number=1000))
print()


t = Timer(lambda: activ_nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: numpy_nn(x))
print (t.timeit(number=1000))
print()


t = Timer(lambda: activ_nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: nn(x))
print (t.timeit(number=1000))

t = Timer(lambda: numpy_nn(x))
print (t.timeit(number=1000))
print()
