import torch
from torch.utils.data import Dataset, DataLoader

class RuleDataset(Dataset):
	def __init__(self, num_relations, relation2rules=None):
		self.rules = list()
		self.num_relations = num_relations
		self.ending_idx = num_relations
		self.padding_idx = num_relations + 1
		if relation2rules != None:
			for rules in relation2rules:
				for rule in rules:
					rule_len = rule[0]
					formatted_rule = [[rule[1 + k] for k in range(rule_len + 1)] + [self.ending_idx], self.padding_idx, rule[-3] + 1e-5]
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

		inputs = torch.LongTensor(inputs)
		target = torch.LongTensor(target)
		weight = torch.Tensor(weight)
		mask = (target != torch.LongTensor(padding_index).unsqueeze(1))

		return inputs, target, mask, weight

def Iterator(dataloader):
	while True:
		for data in dataloader:
			yield data
