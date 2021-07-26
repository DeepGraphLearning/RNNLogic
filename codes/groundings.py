import groundings_cppext as cppgnd
import torch


def use_graph(g):
    cppgnd.init(g.num_node, g.num_relation)
    for e in g.edge_list:
        h, r, t = e.cpu().numpy().tolist()
        cppgnd.add(h, r, t)


def groundings(h, rule, count=False):
    # print("groundings in")
    if not isinstance(rule, list):
        if isinstance(rule, torch.Tensor):
            rule = rule.cpu().numpy().tolist()
        else:
            rule = list(rule)
    if count:
        cppgnd.calc_count(h, rule)
        key = cppgnd.result_pts()
        val = cppgnd.result_cnt()
        # return list(zip(key, val))
        print("groundings out")
        return {k: v for k, v in zip(key, val)}
    else:
        cppgnd.calc(h, rule)
        # print("groundings out")
        return cppgnd.result_pts()
