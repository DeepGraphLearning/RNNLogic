import torch
import rule_sample_cppext as cpprs


def use_graph(g):
    cpprs.init(g.num_node, g.num_relation)
    for e in g.edge_list:
        h, r, t = e.cpu().numpy().tolist()
        cpprs.add(h, r, t)


def sample(r, length_time, num_samples, num_threads=1, samples_per_print=1):
    return cpprs.run(r, length_time, num_samples, num_threads, samples_per_print)


def save(rules, file):
    with open(file, 'w') as fin:
        for rule, (prec, recall) in rules:
            rule = ' '.join(map(str, rule))
            fin.write(f"{rule}\t{prec} {recall}\n")
