import os
import torch
import numpy as np
import json

class RotatE(torch.nn.Module):
    def __init__(self, path):
        super(RotatE, self).__init__()
        self.path = path

        cfg_file = os.path.join(path, 'config.json')
        with open(cfg_file, 'r') as fi:
            cfg = json.load(fi)
        self.emb_dim = cfg['hidden_dim']
        self.gamma = cfg['gamma']
        self.range = (self.gamma + 2.0) / self.emb_dim
        self.num_entities = cfg['nentity']

        eemb_file = os.path.join(path, 'entity_embedding.npy')
        eemb = np.load(eemb_file)
        self.eemb = torch.nn.parameter.Parameter(torch.tensor(eemb))

        remb_file = os.path.join(path, 'relation_embedding.npy')
        remb = np.load(remb_file)
        remb = torch.tensor(remb)
        self.remb = torch.nn.parameter.Parameter(torch.cat([remb, -remb], dim=0))

    def product(self, vec1, vec2):
        re_1, im_1 = torch.chunk(vec1, 2, dim=-1)
        re_2, im_2 = torch.chunk(vec2, 2, dim=-1)

        re_res = re_1 * re_2 - im_1 * im_2
        im_res = re_1 * im_2 + im_1 * re_2

        return torch.cat([re_res, im_res], dim=-1)

    def project(self, vec):
        pi = 3.141592653589793238462643383279
        vec = vec / (self.range / pi)

        re_r = torch.cos(vec)
        im_r = torch.sin(vec)

        return torch.cat([re_r, im_r], dim=-1)

    def diff(self, vec1, vec2):
        diff = vec1 - vec2
        re_diff, im_diff = torch.chunk(diff, 2, dim=-1)
        diff = torch.stack([re_diff, im_diff], dim=0)
        diff = diff.norm(dim=0)
        return diff

    def dist(self, all_h, all_r, all_t):
        h_emb = self.eemb.index_select(0, all_h).squeeze()
        r_emb = self.remb.index_select(0, all_r).squeeze()
        t_emb = self.eemb.index_select(0, all_t).squeeze()

        r_emb = self.project(r_emb)
        e_emb = self.product(h_emb, r_emb)
        dist = self.diff(e_emb, t_emb)

        return dist.sum(dim=-1)
    
    def forward(self, all_h, all_r):
        all_h_ = all_h.unsqueeze(-1).expand(-1, self.num_entities).reshape(-1)
        all_r_ = all_r.unsqueeze(-1).expand(-1, self.num_entities).reshape(-1)
        all_e_ = torch.tensor(list(range(self.num_entities)), dtype=torch.long, device=all_h.device).unsqueeze(0).expand(all_r.size(0), -1).reshape(-1)
        kge_score = self.gamma - self.dist(all_h_, all_r_, all_e_)
        kge_score = kge_score.view(-1, self.num_entities)
        return kge_score

