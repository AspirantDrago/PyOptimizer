from neuronet import Perceptron, ActivePerceptron, PseudoRandomNeuroNet, Neuron
from deductor_parser import DeductorParser
import random

# parser = DeductorParser('test.ded')
# parser.setdocument('Текстовый файл (C:\\Users\\Aleksandr\\Desktop\\PyOptimizer\\test.csv)')
# parser.setneuronet('Нейросеть [2 x 10 x 10 x 1]', parse=True)

nw2 = Perceptron(2, 1, [10, 10])
nw2.setinpnormfrom([[-10, 10], [-10, 10]])
nw2.setoutnormto([[-20, 20]])
nw2.setfilename('neuronet_perceptron.nw')
print(nw2(0, 0))
print(nw2(10, 10))
print()
n = 10000
lst = []
y = 0
for i in range(n):
    x1 = random.random() * 20 - 10
    x2 = random.random() * 20 - 10
    y = x1 + x2
    # y = (x1 + x2 + x1 ** 2 + x2 ** 2 + y) / 2
    # y = i / 100
    lst.append([[x1, x2], [y]])
for _ in range(100):
    err = nw2.learn_bprop(lst, count_batches=None, is_shuffle=False)
    print(nw2(0, 0))
    print(nw2(10, 10))
    print((_ + 1) * n, err)
    nw2.savetofile()
#
# nw = PseudoRandomNeuroNet(2, 1, 20, 5)
# nw.setinpnormfrom([[-10, 10], [-10, 10]])
# nw.setoutnormto([[-20, 120]])
#
# print(nw(0, 0))
# print(nw(10, 10))
# print(nw(0, 0))
# print(nw(10, 10))
#
# n = 1000
# lst = []
# for i in range(n):
#     x1 = random.random() * 20 - 10
#     x2 = random.random() * 20 - 10
#     y = x1 + x2 + x1 ** 2 + x2 ** 2
#     y = i / 10
#     lst.append([[x1, x2], [y]])
# for _ in range(100):
#     err = nw.learn_bprop(lst, count_batches=None, is_shuffle=False)
#     print((_ + 1) * n, err)
