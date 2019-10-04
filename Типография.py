from deductor_parser import DeductorParser
from pprint import pprint

parser = DeductorParser('Типография.ded')
parser.setdocument('Текстовый файл (D:\\Ресурсы\\Исходники\\Типография C#\\KhitSet.csv)')
parser.setneuronet('Нейросеть [4 x 9 x 1]', parse=True)
d = parser.neurodata


pprint(d)
