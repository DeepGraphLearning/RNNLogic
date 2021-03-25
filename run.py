import argparse
import random
from model import KnowledgeGraph, RuleMiner, RNNLogic

def parse_args(args=None):
	parser = argparse.ArgumentParser(
		description='RNNLogic',
		usage='run.py [<args>] [-h | --help]'
	)

	parser.add_argument('--cuda', action='store_true', help='use GPU')

	parser.add_argument('--data_path', type=str, default=None)

	parser.add_argument('--max_rule_length', default=3, type=int)
	parser.add_argument('--num_generated_rules', default=200, type=int)
	parser.add_argument('--num_important_rules', default=10, type=int)
	parser.add_argument('--num_rules_for_test', default=100, type=int)
	parser.add_argument('--prior_weight', default=0.0, type=float)
	parser.add_argument('--num_threads', default=10, type=int)
	parser.add_argument('--iterations', default=10, type=int)

	parser.add_argument('--miner_portion', default=1.0, type=float)

	parser.add_argument('--predictor_learning_rate', default=0.01, type=float)
	parser.add_argument('--predictor_weight_decay', default=0.0005, type=float)
	parser.add_argument('--predictor_temperature', default=100, type=float)
	parser.add_argument('--predictor_H_temperature', default=1.0, type=float)
	parser.add_argument('--predictor_portion', default=1.0, type=float)

	parser.add_argument('--generator_embedding_dim', default=512, type=int)
	parser.add_argument('--generator_hidden_dim', default=256, type=int)
	parser.add_argument('--generator_layers', default=1, type=int)
	parser.add_argument('--generator_batch_size', default=1024, type=int)
	parser.add_argument('--generator_epochs', default=10000, type=int)
	parser.add_argument('--generator_learning_rate', default=0.001, type=float)
	parser.add_argument('--generator_tune_epochs', default=1000, type=int)
	parser.add_argument('--generator_tune_learning_rate', default=0.0001, type=float)

	return parser.parse_args(args)

def main(args):
	graph = KnowledgeGraph(args.data_path)

	miner = RuleMiner(graph)
	miner.mine_logic_rules(args.max_rule_length, args.miner_portion, args.num_threads)

	rnnlogic = RNNLogic(args, graph)
	rnnlogic.m_step(miner.get_logic_rules())

	rnnlogic.train()
	rnnlogic.evaluate()
	
if __name__ == '__main__':
	main(parse_args())
