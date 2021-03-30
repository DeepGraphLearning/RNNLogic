import argparse
import random
from model import KnowledgeGraph, RuleMiner, RNNLogic

def parse_args(args=None):
	parser = argparse.ArgumentParser(
		description='RNNLogic',
		usage='run.py [<args>] [-h | --help]'
	)

	parser.add_argument('--cuda', action='store_true', help='Use GPU.')

	parser.add_argument('--data_path', type=str, default=None, help='Set data path.')

	parser.add_argument('--max_rule_length', default=3, type=int, help='Set the maximum length of logic rules.')
	parser.add_argument('--num_generated_rules', default=200, type=int, help='Set the number of logic rules fed to the reasoning predictor at each training iteration.')
	parser.add_argument('--num_important_rules', default=10, type=int, help='Set the number of high-quality logic rules selected at each iteration.')
	parser.add_argument('--num_rules_for_test', default=100, type=int, help='Set the number of logic rules for evaluation.')
	parser.add_argument('--prior_weight', default=1.0, type=float, help='Set the weight of rule prior when inferring the posterior.')
	parser.add_argument('--num_threads', default=10, type=int, help='Set the number of threads for training.')
	parser.add_argument('--iterations', default=10, type=int, help='Set the number of training iterations.')

	parser.add_argument('--miner_portion', default=1.0, type=float, help='Set the percentage of training triplets used by rule miners.')

	parser.add_argument('--predictor_learning_rate', default=0.01, type=float, help='Set the learning rate for reasoning predictors.')
	parser.add_argument('--predictor_weight_decay', default=0.0005, type=float, help='Set the weight decay for reasoning predictors.')
	parser.add_argument('--predictor_temperature', default=100, type=float, help='Set the annealing temperature of reasoning predictors for prediction.')
	parser.add_argument('--predictor_H_temperature', default=1.0, type=float, help='Set the annealing temperature of reasoning predictors when computing the H score of rules.')
	parser.add_argument('--predictor_portion', default=1.0, type=float, help='Set the percentage of training triplets used by reasoning predictors.')

	parser.add_argument('--generator_embedding_dim', default=512, type=int, help='Set the embedding dimension for rule generators.')
	parser.add_argument('--generator_hidden_dim', default=256, type=int, help='Set the hidden dimension for rule generators.')
	parser.add_argument('--generator_layers', default=1, type=int, help='Set the number of layers for rule generators.')
	parser.add_argument('--generator_batch_size', default=1024, type=int, help='Set the batch size for rule generators.')
	parser.add_argument('--generator_epochs', default=10000, type=int, help='Set the number of training epochs for rule generators.')
	parser.add_argument('--generator_learning_rate', default=0.001, type=float, help='Set learning rate for training rule generators.')
	parser.add_argument('--generator_tune_epochs', default=100, type=int, help='Set the number of tuning epochs for rule generators.')
	parser.add_argument('--generator_tune_learning_rate', default=0.00001, type=float, help='Set learning rate for tuning rule generators.')

	return parser.parse_args(args)

def main(args):
	# Load knowledge graphs.
	graph = KnowledgeGraph(args.data_path)

	# Mine some relational paths from the knowledge graph.
	miner = RuleMiner(graph)
	miner.mine_logic_rules(args.max_rule_length, args.miner_portion, args.num_threads)

	# Pre-train the rule generator on the mined relational paths, 
	# so that the rule generator will not explore useless rules.
	rnnlogic = RNNLogic(args, graph)
	rnnlogic.m_step(miner.get_logic_rules())

	rnnlogic.train()
	rnnlogic.evaluate()
	
if __name__ == '__main__':
	main(parse_args())

# python run.py --data_path ../data/wn18rr --num_generated_rules 200 --num_rules_for_test 200 --num_important_rules 0 --prior_weight 0.01 --cuda
# python run.py --data_path ../data/kinship_325 --num_generated_rules 2000 --num_rules_for_test 200 --num_important_rules 0 --prior_weight 0.01 --cuda
# python run.py --data_path ../data/umls_325 --num_generated_rules 2000 --num_rules_for_test 100 --num_important_rules 0 --prior_weight 0.01 --cuda