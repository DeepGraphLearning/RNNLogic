import os
import torch
import copy
import random
from collections import defaultdict
from dataloader import RuleDataset, Iterator
from torch.utils.data import Dataset, DataLoader
import pyrnnlogic

class KnowledgeGraph:
	def __init__(self, data_path):
		self.pointer = pyrnnlogic.new_knowledge_graph(data_path)

		self.entity2id = dict()
		self.relation2id = dict()
		self.train_triplets = list()
		self.valid_triplets = list()
		self.test_triplets = list()

		with open(os.path.join(data_path, 'entities.dict')) as fi:
			for line in fi:
				eid, entity = line.strip().split('\t')
				self.entity2id[entity] = int(eid)

		with open(os.path.join(data_path, 'relations.dict')) as fi:
			for line in fi:
				rid, relation = line.strip().split('\t')
				self.relation2id[relation] = int(rid)

		with open(os.path.join(data_path, 'train.txt')) as fi:
			for line in fi:
				h, r, t = line.strip().split('\t')
				self.train_triplets.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))

		with open(os.path.join(data_path, 'valid.txt')) as fi:
			for line in fi:
				h, r, t = line.strip().split('\t')
				self.valid_triplets.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))

		with open(os.path.join(data_path, 'test.txt')) as fi:
			for line in fi:
				h, r, t = line.strip().split('\t')
				self.test_triplets.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))

	def get_entity_size(self):
		return len(self.entity2id)

	def get_relation_size(self):
		return len(self.relation2id)

class RuleMiner:
	def __init__(self, graph):
		self.pointer = pyrnnlogic.new_rule_miner(graph.pointer)

	def mine_logic_rules(self, max_length=3, portion=1.0, num_threads=4):
		self.max_length = max_length
		self.num_threads = num_threads
		pyrnnlogic.mine_logic_rules(self.pointer, max_length, portion, num_threads)

	def get_logic_rules(self):
		relation2rules = pyrnnlogic.get_mined_logic_rules(self.pointer)
		return relation2rules

class ReasoningPredictor:
	def __init__(self, graph):
		self.pointer = pyrnnlogic.new_reasoning_predictor(graph.pointer)

	def set_logic_rules(self, relation2rules):
		pyrnnlogic.set_reasoning_predictor(self.pointer, relation2rules)

	def get_logic_rules(self):
		relation2rules = pyrnnlogic.get_assessed_logic_rules(self.pointer)
		return relation2rules

	def train(self, learning_rate=0.01, weight_dacay=0.0005, temperature=100, portion=1.0, num_threads=4):
		pyrnnlogic.train_reasoning_predictor(self.pointer, learning_rate, weight_dacay, temperature, portion, num_threads)

	def evaluate(self, mode="valid", num_threads=4):
		result = pyrnnlogic.test_reasoning_predictor(self.pointer, mode, num_threads)
		return result

	def compute_H_score(self, top_k=0, H_temperature=1, prior_weight=0, portion=1.0, num_threads=4):
		pyrnnlogic.compute_H_score(self.pointer, top_k, H_temperature, prior_weight, portion, num_threads)

class RuleGenerator(torch.nn.Module):
	def __init__(self, num_relations, num_layers, embedding_dim, hidden_dim, cuda=True):
		super(RuleGenerator, self).__init__()
		self.num_relations = num_relations
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = self.num_relations + 2
		self.label_size = self.num_relations + 1
		self.ending_idx = num_relations
		self.padding_idx = self.num_relations + 1
		self.num_layers = num_layers
		self.use_cuda = cuda

		self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
		self.rnn = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
		self.linear = torch.nn.Linear(self.hidden_dim, self.label_size)
		self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

		if cuda:
			self.cuda()

	def zero_state(self, batch_size): 
		state_shape = (self.num_layers, batch_size, self.hidden_dim)
		h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
		if self.use_cuda:
			return (h0.cuda(), c0.cuda())
		else:
			return (h0, c0)

	def forward(self, inputs, relation, hidden):
		embedding = self.embedding(inputs)
		embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
		embedding = torch.cat([embedding, embedding_r], dim=-1)
		outputs, hidden = self.rnn(embedding, hidden)
		logits = self.linear(outputs)
		return logits, hidden

	def loss(self, inputs, target, mask, weight):
		if self.use_cuda:
			inputs = inputs.cuda()
			target = target.cuda()
			mask = mask.cuda()
			weight = weight.cuda()

		hidden = self.zero_state(inputs.size(0))
		logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
		logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
		target = torch.masked_select(target, mask)
		weight = torch.masked_select((mask.t() * weight).t(), mask)
		loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
		return loss

	def log_probability(self, rules):
		if rules == []:
			return []
		with torch.no_grad():
			rules = [rule + [self.ending_idx] for rule in rules]
			max_len = max([len(rule) for rule in rules])
			for k in range(len(rules)):
				rule_len = len(rules[k])
				for i in range(max_len - rule_len):
					rules[k] += [self.padding_idx]
			rules = torch.LongTensor(rules)
			if self.use_cuda:
				rules = rules.cuda()
			inputs = rules[:, :-1]
			target = rules[:, 1:]
			n, l = target.size(0), target.size(1)
			mask = (target != self.padding_idx)
			hidden = self.zero_state(inputs.size(0))
			logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
			logits = torch.log_softmax(logits, -1)
			logits = logits * mask.unsqueeze(-1)
			target = (target * mask).unsqueeze(-1)
			log_prob = torch.gather(logits, -1, target).squeeze(-1) * mask
			log_prob = log_prob.sum(-1)
		return log_prob.data.cpu().numpy().tolist()
		
	def sample(self, relation, num_samples, max_len, temperature=1.0):
		with torch.no_grad():
			rules = torch.zeros([num_samples, max_len + 1]).long() + self.ending_idx
			log_probabilities = torch.zeros([num_samples, max_len + 1])
			head = torch.LongTensor([relation for k in range(num_samples)])
			if self.use_cuda:
				rules = rules.cuda()
				log_probabilities = log_probabilities.cuda()
				head = head.cuda()

			rules[:, 0] = relation
			hidden = self.zero_state(num_samples)

			for pst in range(max_len):
				inputs = rules[:, pst].unsqueeze(-1)
				if self.use_cuda:
					inputs = inputs.cuda()
				logits, hidden = self.forward(inputs, head, hidden)
				logits *= temperature
				log_probability = torch.log_softmax(logits.squeeze(1), dim=-1)
				probability = torch.softmax(logits.squeeze(1), dim=-1)
				sample = torch.multinomial(probability, 1)
				log_probability = log_probability.gather(1, sample)

				mask = (rules[:, pst] != self.ending_idx)
				
				rules[mask, pst + 1] = sample.squeeze(-1)[mask]
				log_probabilities[mask, pst + 1] = log_probability.squeeze(-1)[mask]

			length = (rules != self.ending_idx).sum(-1).unsqueeze(-1) - 1
			formatted_rules = torch.cat([length, rules], dim=1)

			log_probabilities = log_probabilities.sum(-1)

		formatted_rules = formatted_rules.data.cpu().numpy().tolist()
		log_probabilities = log_probabilities.data.cpu().numpy().tolist()
		for k in range(num_samples):
			formatted_rules[k].append(log_probabilities[k])

		rule_set = set([tuple(rule) for rule in formatted_rules])
		formatted_rules = [list(rule) for rule in rule_set]

		return formatted_rules

	def next_relation_log_probability(self, seq):
		inputs = torch.LongTensor([seq])
		relation = torch.LongTensor([seq[0]])
		if self.use_cuda:
			inputs = inputs.cuda()
			relation = relation.cuda()
		hidden = self.zero_state(1)
		logits, hidden = self.forward(inputs, relation, hidden)
		log_prob = torch.log_softmax(logits[0, -1, :] * 5, dim=-1).data.cpu().numpy().tolist()
		return log_prob

	def train_model(self, iterator, num_epoch=10000, lr=1e-3, print_epoch=100):
		opt = torch.optim.Adam(self.parameters(), lr=lr)
		sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr/10)

		cum_loss = 0
		for epoch in range(1, num_epoch + 1):
			epoch += 1

			batch = next(iterator)
			inputs, target, mask, weight = batch

			loss = self.loss(inputs, target, mask, weight)
			opt.zero_grad()
			loss.backward()
			opt.step()
			sch.step()

			cum_loss += loss.item()

			if epoch % print_epoch == 0:
				lr_str = "%.2e" % (opt.param_groups[0]['lr'])
				print(f"train_generator #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
				cum_loss = 0
			if epoch == num_epoch:
				break

	def beam_search(self, relation, num_samples, max_len):
		max_len += 1
		with torch.no_grad():
			found_rules = []
			prev_rules = [[[relation], 0]]
			for k in range(max_len):
				print(f"k = {k} |prev| = {len(prev_rules)}")
				current_rules = list()
				for _i, (rule, score) in enumerate(prev_rules):
					assert rule[-1] != self.ending_idx
					log_prob = self.next_relation_log_probability(rule)
					for i in (range(self.label_size) if (k + 1) != max_len else [self.ending_idx]):
						new_rule = rule + [i]
						new_score = score + log_prob[i]
						(current_rules if i != self.ending_idx else found_rules).append((new_rule, new_score))
					
				prev_rules = sorted(current_rules, key=lambda x:x[1], reverse=True)[:num_samples]
				found_rules = sorted(found_rules, key=lambda x:x[1], reverse=True)[:num_samples]

			print(f"beam_search |rules| = {len(found_rules)}")
			ret = [[len(rule) - 2] + rule[0:-1] + [score] for rule, score in found_rules]
			return ret

class RNNLogic:
	def __init__(self, args, graph):
		self.args = args
		self.graph = graph
		self.num_relations = graph.get_relation_size()
		self.predictor = ReasoningPredictor(graph)
		self.generator = RuleGenerator(self.num_relations, args.generator_layers, args.generator_embedding_dim, args.generator_hidden_dim, args.cuda)

	# Generate logic rules by sampling.
	def generate_rules(self):
		relation2rules = list()
		for r in range(self.num_relations):
			rules = self.generator.sample(r, self.args.num_generated_rules, self.args.max_rule_length)
			relation2rules.append(rules)
		return relation2rules

	# Generate optimal logic rules by beam search.
	def generate_best_rules(self):
		relation2rules = list()
		for r in range(self.num_relations):
			rules = self.generator.beam_search(r, self.args.num_rules_for_test, self.args.max_rule_length)
			relation2rules.append(rules)
		return relation2rules

	# Update the reasoning predictor with generated logic rules.
	def update_predictor(self, relation2rules):
		self.predictor.set_logic_rules(relation2rules)
		self.predictor.train(self.args.predictor_learning_rate, self.args.predictor_weight_decay, self.args.predictor_temperature, self.args.predictor_portion, self.args.num_threads)

	# E-step: Infer the high-quality logic rules.
	def e_step(self):
		self.predictor.compute_H_score(self.args.num_important_rules, self.args.predictor_H_temperature, self.args.prior_weight, self.args.predictor_portion, self.args.num_threads)
		return self.predictor.get_logic_rules()

	# M-step: Update the rule generator with logic rules.
	def m_step(self, relation2rules, tune=False):
		dataset = RuleDataset(self.num_relations, relation2rules)
		dataloader = DataLoader(dataset, batch_size=self.args.generator_batch_size, shuffle=True, num_workers=1, collate_fn=RuleDataset.collate_fn)
		iterator = Iterator(dataloader)
		if not tune:
			self.generator.train_model(iterator, num_epoch=self.args.generator_epochs, lr=self.args.generator_learning_rate)
		else:
			self.generator.train_model(iterator, num_epoch=self.args.generator_tune_epochs, lr=self.args.generator_tune_learning_rate)

	def train(self):
		all_high_quality_rules = [[] for r in range(self.num_relations)]
		
		for iteration in range(self.args.iterations):

			# Generate a set of logic rules and update the reasoning predictor for reasoning.
			relation2rules = self.generate_rules()
			self.update_predictor(relation2rules)
			mr, mrr, hit1, hit3, hit10 = self.predictor.evaluate('valid', self.args.num_threads)
			print("Valid | MR: {:.6f}, MRR: {:.6f}, Hit@1: {:.6f}, Hit@3: {:.6f}, Hit@10: {:.6f}.".format(mr, mrr, hit1, hit3, hit10))

			# E-step: Identify a subset of high-quality logic rules based on posterior inference.
			high_quality_rules = self.e_step()

			# M-step: Improve the rule generator with the high-quality rules from the E-step.
			self.m_step(high_quality_rules, tune=True)

			for r in range(self.num_relations):
				all_high_quality_rules[r] += high_quality_rules[r]

		self.m_step(all_high_quality_rules)

	def evaluate(self):
		relation2rules = self.generate_best_rules()
		self.predictor.set_logic_rules(relation2rules)
		self.predictor.train(self.args.predictor_learning_rate, self.args.predictor_weight_decay, self.args.predictor_temperature, self.args.predictor_portion, self.args.num_threads)
		mr, mrr, hit1, hit3, hit10 = self.predictor.evaluate("test", self.args.num_threads)
		print("Test | MR: {:.6f}, MRR: {:.6f}, Hit@1: {:.6f}, Hit@3: {:.6f}, Hit@10: {:.6f}.".format(mr, mrr, hit1, hit3, hit10))
