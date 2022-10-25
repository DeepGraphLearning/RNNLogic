import torch
from torch.utils.data import Dataset
from torch_scatter import scatter
import numpy as np
import os
import random
from easydict import EasyDict

class KnowledgeGraph(object):
    def __init__(self, data_path):
        self.data_path = data_path

        self.entity2id = dict()
        self.relation2id = dict()
        self.id2entity = dict()
        self.id2relation = dict()

        with open(os.path.join(data_path, 'entities.dict')) as fi:
            for line in fi:
                id, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id)
                self.id2entity[int(id)] = entity

        with open(os.path.join(data_path, 'relations.dict')) as fi:
            for line in fi:
                id, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id)
                self.id2relation[int(id)] = relation

        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        
        self.train_facts = list()
        self.valid_facts = list()
        self.test_facts = list()
        self.hr2o = dict()
        self.hr2oo = dict()
        self.hr2ooo = dict()
        self.relation2adjacency = [[[], []] for k in range(self.relation_size)]
        self.relation2ht2index = [dict() for k in range(self.relation_size)]
        self.relation2outdegree = [[0 for i in range(self.entity_size)] for k in range(self.relation_size)]

        with open(os.path.join(data_path, "train.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.train_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)
                
                if hr_index not in self.hr2o:
                    self.hr2o[hr_index] = list()
                self.hr2o[hr_index].append(t)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

                self.relation2adjacency[r][0].append(t)
                self.relation2adjacency[r][1].append(h)

                ht_index = self.encode_ht(h, t)
                assert ht_index not in self.relation2ht2index[r]
                index = len(self.relation2ht2index[r])
                self.relation2ht2index[r][ht_index] = index

                self.relation2outdegree[r][t] += 1

        with open(os.path.join(data_path, "valid.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.valid_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2oo:
                    self.hr2oo[hr_index] = list()
                self.hr2oo[hr_index].append(t)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

        with open(os.path.join(data_path, "test.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.test_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)

                if hr_index not in self.hr2ooo:
                    self.hr2ooo[hr_index] = list()
                self.hr2ooo[hr_index].append(t)

        for r in range(self.relation_size):
            index = torch.LongTensor(self.relation2adjacency[r])
            value = torch.ones(index.size(1))
            self.relation2adjacency[r] = [index, value]

            self.relation2outdegree[r] = torch.LongTensor(self.relation2outdegree[r])

        print("Data loading | DONE!")

    def encode_hr(self, h, r):
        return r * self.entity_size + h

    def decode_hr(self, index):
        h, r = index % self.entity_size, index // self.entity_size
        return h, r

    def encode_ht(self, h, t):
        return t * self.entity_size + h

    def decode_ht(self, index):
        h, t = index % self.entity_size, index // self.entity_size
        return h, t

    def get_updated_adjacency(self, r, edges_to_remove):
        if edges_to_remove == None:
            return None
        index = self.relation2sparse[r][0]
        value = self.relation2sparse[r][1]
        mask = (index.unsqueeze(1) == edges_to_remove.unsqueeze(-1))
        mask = mask.all(dim=0).any(dim=0)
        mask = ~mask
        index = index[:, mask]
        value = value[mask]
        return [index, value]

    def grounding(self, h, r, rule, edges_to_remove):
        device = h.device
        with torch.no_grad():
            x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1).unsqueeze(-1)
            if device.type == "cuda":
                x = x.cuda(device)
            for r_body in rule:
                if r_body == r:
                    x = self.propagate(x, r_body, edges_to_remove)
                else:
                    x = self.propagate(x, r_body, None)
        return x.squeeze(-1).transpose(0, 1)

    def propagate(self, x, relation, edges_to_remove=None):
        device = x.device
        node_in = self.relation2adjacency[relation][0][1]
        node_out = self.relation2adjacency[relation][0][0]
        if device.type == "cuda":
            node_in = node_in.cuda(device)
            node_out = node_out.cuda(device)

        message = x[node_in]
        E, B, D = message.size()

        if edges_to_remove == None:
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))
        else:
            # message: edge * batch * dim
            message = message.view(-1, D)
            bias = torch.arange(B)
            if device.type == "cuda":
                bias = bias.cuda(device)
            edges_to_remove = edges_to_remove * B + bias
            message[edges_to_remove] = 0
            message = message.view(E, B, D)
            x = scatter(message, node_out, dim=0, dim_size=x.size(0))

        return x

class TrainDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        self.r2instances = [[] for r in range(self.graph.relation_size)]
        for h, r, t in self.graph.train_facts:
            self.r2instances[r].append((h, r, t))

        self.make_batches()

    def make_batches(self):
        for r in range(self.graph.relation_size):
            random.shuffle(self.r2instances[r])

        self.batches = list()
        for r, instances in enumerate(self.r2instances):
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])
        target = torch.zeros(len(data), self.graph.entity_size)
        edges_to_remove = []
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2o[hr_index])
            target[k][t_index] = 1

            ht_index = self.graph.encode_ht(h, t)
            edge = self.graph.relation2ht2index[r][ht_index]
            edges_to_remove.append(edge)
        edges_to_remove = torch.LongTensor(edges_to_remove)

        return all_h, all_r, all_t, target, edges_to_remove

class ValidDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        facts = self.graph.valid_facts

        r2instances = [[] for r in range(self.graph.relation_size)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2oo[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

class TestDataset(Dataset):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        facts = self.graph.test_facts

        r2instances = [[] for r in range(self.graph.relation_size)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            random.shuffle(instances)
            for k in range(0, len(instances), self.batch_size):
                start = k
                end = min(k + self.batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.graph.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.graph.encode_hr(h, r)
            t_index = torch.LongTensor(self.graph.hr2ooo[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

class RuleDataset(Dataset):
    def __init__(self, num_relations, input):
        self.rules = list()
        self.num_relations = num_relations
        self.ending_idx = num_relations
        self.padding_idx = num_relations + 1
        
        if type(input) == list:
            rules = input
        elif type(input) == str:
            rules = list()
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule[0:-1]] + [float(rule[-1]) * 1000]
                    rules.append(rule)
        
        self.rules = []
        for rule in rules:
            rule_len = len(rule)
            formatted_rule = [rule[0:-1] + [self.ending_idx], self.padding_idx, rule[-1] + 1e-5]
            self.rules.append(formatted_rule)
    
    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx]

    @staticmethod
    def collate_fn(data):
        inputs = [item[0][0:len(item[0])-1] for item in data]
        target = [item[0][1:len(item[0])] for item in data]
        weight = [float(item[-1]) for item in data]
        max_len = max([len(_) for _ in inputs])
        padding_index = [int(item[-2]) for item in data]

        for k in range(len(data)):
            for i in range(max_len - len(inputs[k])):
                inputs[k].append(padding_index[k])
                target[k].append(padding_index[k])

        inputs = torch.tensor(inputs, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        weight = torch.tensor(weight)
        mask = (target != torch.tensor(padding_index, dtype=torch.long).unsqueeze(1))

        return inputs, target, mask, weight

def Iterator(dataloader):
    while True:
        for data in dataloader:
            yield data