# RNNLogic

This is an implementation of the [RNNLogic](https://arxiv.org/abs/2010.04029) model for knowledge graph reasoning. 

## Introduction

RNNLogic focuses on knowledge graphs, which are collections of real-world facts, with each fact represented by a (h,r,t)​-triplet. As collecting facts is expensive, knowledge graphs are imcomplete, and thus predicting missing facts in a knowledge graph is an important problem with growing attention. Such a problem is known as **knowledge graph reasoning**.

RNNLogic solves knowledge graph reasoning by learning **logic rules**, which have been proved to improve the *interpretability* and *precision* of reasoning. To do that, RNNLogic employs a **rule generator** and a **reasoning predictor**. The rule generator is parameterized by a RNN, which is able to model and generate chain-like rules. The reasoning predictor follows stochastic logic programming, which uses a set of logic rules as input to predict the answers of queries. Given a query, the rule generator generates a set of logic rules, which are fed into the reasoning predictor. The rule generator further applies the logic rules to the existing knowledge graph for predicting the answer.

<img src="./figures/workflow.png" alt="workflow" img width="50%" />

To optimize the reasoning predictor and the rule generator, we propose an **EM-based algorithm**. At each iteration, the algorithm starts with generating a set of logic rules, which are fed into the reasoning predictor and we further update the reasoning predictor based on the training queries and answers.

<img src="./figures/pre-step.png" alt="pre-step" img width="50%" />

Then in the E-step, a set of high-quality logic rules are selected from all the generated logic rules according to their posterior probabilities.

<img src="./figures/e-step.png" alt="e-step" img width="50%" />

Finally in the M-step, the rule generator is updated to be consistent with the high-logic logic rules identified in the E-step.

<img src="./figures/m-step.png" alt="m-step" img width="50%" />

## Data
We provide four datasets for knowledge graph reasoning, and these datasets are FB15k-237, WN18RR, Kinship, UMLS. For FB15k-237 and WN18RR, there are standard splits for the training/validation/test sets. For Kinship and UMLS, we split the training/validation/test sets by ourselves, and the details are available in our paper.

## Usage

We provide two versions of RNNLogic implemetations.

### Version 1

In the folder **codes_toy**, we provide a toy implementation, which only implements the model *RNNLogic w/o emb*. This toy implementation is easy to understand, which can be used to reproduce the results of *RNNLogic w/o emb* reported in the paper.

To run this code, please first go to the directory *codes_toy/pyrnnlogiclib* and run the following command:
```
python setup.py install
```
After that, you can go back to the main directory and use the following commands to do knowledge graph reasoning:
```
python run.py --data_path ../data/wn18rr --num_generated_rules 200 --num_rules_for_test 200 --num_important_rules 0 --prior_weight 0.01 --cuda
python run.py --data_path ../data/kinship --num_generated_rules 2000 --num_rules_for_test 200 --num_important_rules 0 --prior_weight 0.01 --cuda
python run.py --data_path ../data/umls --num_generated_rules 2000 --num_rules_for_test 100 --num_important_rules 0 --prior_weight 0.01 --cuda
```

### Version 2

In the folder *codes*, we provide the full implementation of both *RNNLogic w/o emb* and *RNNLogic with emb*. The codes can be used to reproduce the results of *RNNLogic with emb* reported in the paper. Note that in order to speed up training, this implementation trains a separate model for each relation, which allows us to deal with all the relations in parallel.

To run this code, please edit the script *codes/run.py* and use this script for model training.

## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!
```
@inproceedings{qu2020rnnlogic,
  title={RNNLogic: Learning Logic Rules for Reasoning on Knowledge Graphs},
  author={Qu, Meng and Chen, Junkun and Xhonneux, Louis-Pascal and Bengio, Yoshua and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
