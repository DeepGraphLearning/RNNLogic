import time

from knowledge_graph_utils import *
from metrics import Metrics
from model_rnnlogic import RNNLogic
from rotate import RotatE

DATASET = "dataset/kinship_325/"

kinship = load_dataset(f"{DATASET}")

# kinship['valid'] = list(filter(lambda x : x[1] == 0, kinship['valid']))[:1]
# kinship['test'] = ""


rotate_kinship = RotatE(kinship, pretrained=f"{DATASET}/RotatE_500")
res = rotate_kinship.infer(filter(lambda x: x[1] == 0, kinship['valid']), valid=True)

print(Metrics.pretty(Metrics.summary(res)))

import rule_sample

rule_sample.use_graph(dataset_graph(kinship, 'train'))
rules = rule_sample.sample(0, {3: 10}, 1000, num_threads=12, samples_per_print=100)
rule_sample.save(rules, "rules.txt")

# exit()


model = RNNLogic(kinship,
                 {
                     'rotate_pretrained': None,
                     'max_rules': 1000,
                     'max_rule_len': 3,
                     'max_beam_rules': 2000,
                     'num_em_epoch': 2,
                     'predictor_batch_size': 8,
                     'predictor_lr': 1e-4,
                     'predictor_num_epoch': 100000,
                     'use_neg_rules': True,
                     'max_pgnd_rules': 0
                 })

for name, param in model.named_parameters():
    print(f"Model Parameter: {name} ({param.type()}:{param.size()})")

model.train_model(1, rule_file=f"{DATASET}/Rules/rules_1.txt")

exit()

exit()


def cntwrg(x):
    return ((x / x.size(0)).abs() > 1e-4).sum().item()


import rotate
from random import randint

N = 100000
MA = 1000
MB = 20000
D = 2000

a = torch.randn(MA, D * 2).cuda()
b = torch.randn(MB, D * 2).cuda()
pa = torch.tensor([randint(0, MA - 1) for _ in range(N)]).long().cuda()
pb = torch.tensor([randint(0, MB - 1) for _ in range(N)]).long().cuda()

c = torch.randn(N).cuda()

fun0 = rotate.RotatECompare_Force()
fun1 = rotate.RotatECompare()


def test(fun, a, b, pa, pb, c):
    a = a.clone()
    b = b.clone()
    a.requires_grad_()
    b.requires_grad_()

    d = fun.apply(a, b, pa, pb)
    (d * c).sum().backward()

    return d, a.grad, b.grad


# t = time.time()
# d0, a0, b0 = test(fun0, a, b, pa, pb, c)
# print(time.time() - t)

t = time.time()
d1, a1, b1 = test(fun1, a, b, pa, pb, c)
print(time.time() - t)

# print(cntwrg(d0-d1), cntwrg(a0-a1), cntwrg(b0-b1))


exit()

import rotate as rotate_dist

N = 1000
D = 500
a = torch.randn(N, D * 2).cuda()
x = torch.randn(D * 2).cuda()

fun0 = rotate_dist.RotatEDist_Force()
fun1 = rotate_dist.RotatEDist_Force2()
fun2 = rotate_dist.RotatEDist()

c = torch.randn(N).cuda()


def test(fun, x, a, c):
    x = x.clone()
    a = a.clone()
    x.requires_grad_()
    a.requires_grad_()

    dist = fun.apply(x, a)
    (dist * c).sum().backward()

    return dist, x.grad, a.grad


t = time.time()
d0, x0, a0 = test(fun0, x, a, c)
print(time.time() - t)

# t = time.time()
# d1, x1, a1 = test(fun1, x, a, c)
# print(time.time() - t)

t = time.time()
d2, x2, a2 = test(fun2, x, a, c)
print(time.time() - t)

# print(x0)
# print(x1)
# print(x2)


print(cntwrg(d0 - d2), cntwrg(x0 - x2), cntwrg(a0 - a2))

exit()

umls = load_graph("dataset/umls_325")
kinship = load_graph("dataset/kinship_325")
fb = load_graph("dataset/fb15k237")

import groundings

groundings.use_graph(umls)
print(groundings.groundings(0, [0]))
print(groundings.groundings(0, [0, 57, 49], count=True))

import rule_sample

rule_sample.use_graph(fb)

t = time.time()
rules = rule_sample.sample(0, {3: 10}, 1000, num_threads=1, samples_per_print=100)
t1 = time.time() - t

t = time.time()
rules = rule_sample.sample(0, {3: 10}, 1000, num_threads=12, samples_per_print=100)
t2 = time.time() - t

print(t1, t2)
