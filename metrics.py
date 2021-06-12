import torch

from knowledge_graph_utils import list2mask


class Metrics:

    def __init__(self, N):
        self.N = N
        self.pre_mr = [0]
        self.pre_mrr = [0]
        self.pre_h1 = [0]
        self.pre_h3 = [0]
        self.pre_h10 = [0]
        for i in range(1, N + 10):
            self.pre_mr.append(self.pre_mr[-1] + i)
            self.pre_mrr.append(self.pre_mrr[-1] + 1.0 / i)
            self.pre_h1.append(self.pre_h1[-1] + (1 if i <= 1 else 0))
            self.pre_h3.append(self.pre_h3[-1] + (1 if i <= 3 else 0))
            self.pre_h10.append(self.pre_h10[-1] + (1 if i <= 10 else 0))

    def apply(self, score, answer, t):
        score = score.clone()
        score[torch.isnan(score)] = 0
        if isinstance(answer, list) or not isinstance(answer, torch.BoolTensor):
            answer = list2mask(answer, score.size(-1))
        answer = answer.cuda()
        # print("apply", mask2list(answer), t)
        with torch.no_grad():
            incorrect = score[~answer.bool()]
            rankl = (incorrect > score[t]).sum().item()  # + 1
            rankr = (incorrect >= score[t]).sum().item() + 1
            mr = (self.pre_mr[rankr] - self.pre_mr[rankl]) / (rankr - rankl)
            mrr = (self.pre_mrr[rankr] - self.pre_mrr[rankl]) / (rankr - rankl)
            h1 = (self.pre_h1[rankr] - self.pre_h1[rankl]) / (rankr - rankl)
            h3 = (self.pre_h3[rankr] - self.pre_h3[rankl]) / (rankr - rankl)
            h10 = (self.pre_h10[rankr] - self.pre_h10[rankl]) / (rankr - rankl)
        return 1, mr, mrr, h1, h3, h10

    @staticmethod
    def num():
        return 6

    @staticmethod
    def zero_value():
        return (0,) * Metrics.num()

    @staticmethod
    def init_value():
        return (-1,) * Metrics.num()

    @staticmethod
    def merge(a, b):
        c = [a[0] + b[0]]
        for x, y in zip(a[1:], b[1:]):
            c.append((x * a[0] + y * b[0]) / (a[0] + b[0]))
        return tuple(c)

    @staticmethod
    def summary(results):
        a = list(Metrics.zero_value())
        for r in results:
            a = Metrics.merge(a, r)
        return a

    @staticmethod
    def pretty(a):
        return {
            'num': int(a[0]),
            'mr': a[1],
            'mrr': a[2],
            'h1': a[3],
            'h3': a[4],
            'h10': a[5],
        }

    @staticmethod
    def format(result):
        s = ""
        for k, v in Metrics.pretty(result).items():
            if k == 'num':
                continue
            s += k + ":"
            s += "%.4lf " % v
        return s
