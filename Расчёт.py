from neuronet import ActivePerceptron
from deductor_parser import DeductorParser
from optimizer import OptimizerDE, OptimizerAnnealing
# import timeit

parser = DeductorParser('1.ded')
parser.setdocument(r'Текстовый файл (D:\Олимпиады\data.csv)')
parser.setneuronet(r'Нейросеть [3 x 3 x 1]', parse=True)

nw = ActivePerceptron()
nw.loader(parser.neurodata)


opt = OptimizerDE()
# opt = OptimizerAnnealing()
opt.model = nw
opt.func = lambda x, y: y[0]
# opt.initialize([50, -100000000, 39])
# opt.initialize([[80, 130, 39], [81, 131, 39.2], [82, 132, 39.3], [83, 133, 39.4]])
opt.var = [True, True, True]
opt.print_log = True
opt.intervals = parser.neurodata['inp_norm']
opt.full_width_variation(1, count=3)
opt.distrib = opt.distrib_normal
opt.initialize_random(4)
opt.epoches = True
opt.epoches_limit = 1000
#opt.method = opt.method_boltzman
opt.t0 = 1
opt.idle = True
opt.idle_limit = 3
opt.tol = False
opt.tol_limit = 0.001

# opt.cond_limit = lambda x, y, e: e <= 83
# opt.limits_in.append(lambda x: x[2] >= 35.6)
# opt.limits_in.append(lambda x: x[2] - int(x[2]) < 0.5)
# opt.limits_out.append(lambda x: x[0] - int(x[0]) < 0.5)
# opt.limits_out.append(lambda x: x[0] < 85)

opt.start()
opt.save()


print(opt.val)
print(opt.out)
print(opt.error)
