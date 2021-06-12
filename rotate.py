from collections import defaultdict

import rotate_compare_cppext
import rotate_dist_cppext
import torch

from reasoning_model import ReasoningModel


class RotatE(ReasoningModel):

    def __init__(self, dataset, pretrained=None):
        # load pretrained rotate
        super(RotatE, self).__init__()

        E = dataset['E']
        R = dataset['R']
        self.E = E
        self.R = R

        self.answer = defaultdict(lambda: [])
        self.answer_test = defaultdict(lambda: [])

        for item in ['train', 'test', 'valid']:
            for (h, r, t) in dataset[item]:
                if item != 'test':
                    self.answer[(h, r)].append(t)

                self.answer_test[(h, r)].append(t)

        if pretrained is None:
            self.gamma = 0.0
            self.embed_dim = 1
            self.embed_range = 1.0
            self.entity_embed = torch.zeros(E, 2).float()
            self.relation_embed = torch.zeros(R, 1).float()
        else:
            import numpy
            import json
            config = json.load(open(f"{pretrained}/config.json"))
            self.gamma = config['gamma']
            self.embed_dim = config['hidden_dim']
            self.embed_range = (self.gamma + 2.0) / self.embed_dim
            self.entity_embed = torch.tensor(numpy.load(f"{pretrained}/entity_embedding.npy"))
            relation_embed = torch.tensor(numpy.load(f"{pretrained}/relation_embedding.npy"))
            self.relation_embed = (torch.cat([relation_embed, -relation_embed], dim=0))

        # pi = 3.141592653589793238462643383279
        # self.relation_embed = self.relation_embed / self.embed_range * pi

        self.entity_embed = self.entity_embed.cuda()
        self.relation_embed = self.relation_embed.cuda()
        # self._tmp = torch.nn.Parameter(torch.zeros(1))
        self.cuda()

    def _attatch_empty_relation(self):
        return torch.cat([self.relation_embed, torch.zeros(self.embed_dim).cuda().unsqueeze(0)], dim=0)

    @staticmethod
    def dist(*args):
        return RotatEDist.apply(*args)

    @staticmethod
    def compare(*args):
        return RotatECompare.apply(*args)

    def embed(self, h_embed, r_embed):
        if isinstance(h_embed, int):
            h_embed = self.entity_embed.index_select(0, torch.tensor(h_embed).cuda()).squeeze().cuda()
        if isinstance(r_embed, int):
            r_embed = self.relation_embed.index_select(0, torch.tensor(r_embed).cuda()).squeeze().cuda()

        re_h, im_h = torch.chunk(h_embed, 2, dim=-1)

        pi = 3.141592653589793238462643383279
        r_embed = r_embed / (self.embed_range / pi)
        re_r = torch.cos(r_embed)
        im_r = torch.sin(r_embed)

        re_res = re_h * re_r - im_h * im_r
        im_res = re_h * im_r + im_h * re_r

        return torch.cat([re_res, im_res], dim=-1)

    def infer(self, infer_tris, valid=False, graph=None):
        self.entity_embed = self.entity_embed.cuda()
        self.relation_embed = self.relation_embed.cuda()

        results = []
        metrics = self.metrics
        with torch.no_grad():
            for i, (h, r, t) in enumerate(infer_tris):
                score = self.gamma - self.dist(self.embed(h, r), self.entity_embed)
                answer = (self.answer if valid else self.answer_test)[(h, r)]
                results.append(metrics.apply(score, answer, t))
            # print(h, r, t, answer)
            # print(score)

        return results


class RotatECompare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, pa, pb):
        a = a.contiguous()
        b = b.contiguous()
        pa = pa.contiguous()
        pb = pb.contiguous()
        # print("RotatE compare", a.size(), b.size())
        # print(pa.min().item(), pa.max().item())
        # print(pb.min().item(), pb.max().item())

        ctx.save_for_backward(a, b, pa, pb)
        return rotate_compare_cppext.forward(a, b, pa, pb)

    @staticmethod
    def backward(ctx, ogd):
        a, b, pa, pb = ctx.saved_tensors
        return rotate_compare_cppext.backward(a, b, pa, pb, ogd)


class RotatECompare_Force:
    @staticmethod
    def apply(a, b, pa, pb):
        # print("Warning: Force version used")
        a = a.contiguous()
        b = b.contiguous()
        pa = pa.contiguous()
        pb = pb.contiguous()
        a = a.index_select(0, pa)
        b = b.index_select(0, pb)
        dist = a - b
        re, im = torch.chunk(dist, 2, dim=-1)
        dist = torch.stack([re, im], dim=-1)
        dist = dist.norm(dim=-1).sum(dim=-1)
        # if pa.size(0) == 1030:
        # 	exit()
        return dist


class RotatEDist(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        x = x.contiguous()
        a = a.contiguous()
        dist = rotate_dist_cppext.forward(x, a)
        ctx.save_for_backward(x, a)
        return dist

    @staticmethod
    def backward(ctx, outgrad_dist):
        # print(outgrad_dist)
        x, a = ctx.saved_tensors
        ingrad_x, ingrad_a = rotate_dist_cppext.backward(x, a, outgrad_dist)
        return ingrad_x, ingrad_a


class RotatEDist_Force:
    @staticmethod
    def apply(x, a):
        print("Warning: Force version used")
        a = x - a
        re, im = torch.chunk(a, 2, dim=-1)
        a = torch.stack([re, im], dim=-1)
        dist = a.norm(dim=-1).sum(dim=-1)
        # print(dist.size())
        return dist


class RotatEDist_Force2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        x = x.detach()
        a = a.detach()
        tmp = x - a
        re, im = torch.chunk(tmp, 2, dim=-1)
        tmp = torch.stack([re, im], dim=-1)
        dist = tmp.norm(dim=-1).sum(dim=-1)
        ctx.save_for_backward(x, a)
        # print(dist.size())
        return dist

    @staticmethod
    def backward(ctx, o):
        x, a = ctx.saved_tensors
        # print("enter backward", a.size(), x.size())

        are, aim = torch.chunk(a, 2, dim=-1)
        xre, xim = torch.chunk(x, 2, dim=-1)

        gxre = torch.zeros_like(xre).cuda()
        gxim = torch.zeros_like(xim).cuda()
        gare = torch.zeros_like(are).cuda()
        gaim = torch.zeros_like(aim).cuda()

        n = are.size(0)
        d = are.size(1)

        # print(n, d)

        for i in range(n):
            for j in range(d):
                re = xre[j] - are[i][j]
                im = xim[j] - aim[i][j]
                dis = (re ** 2 + im ** 2) ** 0.5
                # print("%d %d %.4lf %.4lf" % (i,j,dis,o[i]))
                gxre[j] += re * o[i] / dis
                gxim[j] += im * o[i] / dis
                gare[i][j] = -re * o[i] / dis
                gaim[i][j] = -im * o[i] / dis

        gx = torch.cat([gxre, gxim], dim=-1)
        ga = torch.cat([gare, gaim], dim=-1)

        # print(gx.size(), ga.size())
        return gx, ga
