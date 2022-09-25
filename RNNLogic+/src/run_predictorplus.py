import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset
from predictors import PredictorPlus
from utils import load_config, save_config, set_logger, set_seed
from trainer import TrainerPredictor
import comm

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--config', default='../predictor.yaml', type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args(args)

def main(args):
    cfgs = load_config(args.config)
    cfg = cfgs[0]

    if cfg.save_path is None:
        cfg.save_path = os.path.join('/home/qumeng/scratch/rnnlogic/outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    
    if cfg.save_path and not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    
    save_config(cfg, cfg.save_path)

    set_logger(cfg.save_path)
    set_seed(cfg.seed)

    graph = KnowledgeGraph(cfg.data.data_path)
    train_set = TrainDataset(graph, cfg.data.batch_size)
    valid_set = ValidDataset(graph, cfg.data.batch_size)
    test_set = TestDataset(graph, cfg.data.batch_size)

    predictor = PredictorPlus(graph, **cfg.predictor.model)
    predictor.set_rules(cfg.data.rule_file)
    optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)

    solver = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, gpus=cfg.gpus)
    best_valid_mrr = 0.0
    test_mrr = 0.0
    for k in range(cfg.num_iters):
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Iteration: {}/{}'.format(k + 1, cfg.num_iters))
            logging.info('-------------------------')
        
        solver.train(**cfg.predictor.train)
        valid_mrr_iter = solver.evaluate('valid', expectation=cfg.predictor.eval.expectation)
        test_mrr_iter = solver.evaluate('test', expectation=cfg.predictor.eval.expectation)
        if valid_mrr_iter > best_valid_mrr:
            best_valid_mrr = valid_mrr_iter
            test_mrr = test_mrr_iter
            solver.save(os.path.join(cfg.save_path, 'predictor.pt'))

if __name__ == '__main__':
    main(parse_args())