# RNNLogic

This is an implementation of the [RNNLogic](https://openreview.net/forum?id=tGZu6DlbreV) model for knowledge graph reasoning. 

Note that for now this repository only provides the codes of *RNNLogic w/o emb*, and the complete version will be released soon together with a graph reasoning package implemented by our research group **MilaGraph**. Stay tuned!

Also, the current codes are lack of enough comments and **we are keep updating the repository**.

## Introduction

RNNLogic focuses on knowledge graphs, which are collections of real-world facts, with each fact represented by a (h,r,t)​-triplet. As collecting facts is expensive, knowledge graphs are imcomplete, and thus predicting missing facts in a knowledge graph is an important problem with growing attention. Such a problem is known as **knowledge graph reasoning**.

RNNLogic solves knowledge graph reasoning by learning **logic rules**, which have been proved to improve the *interpretability* and *precision* of reasoning. To do that, RNNLogic employs a **rule generator** and a **reasoning predictor**. The rule generator is parameterized by a RNN, which is able to model and generate chain-like rules. The reasoning predictor follows stochastic logic programming, which uses a set of logic rules as input to predict the answers of queries.

Formally, for each query-answer pair $q=(h, r, ?)$ and $a = t$, RNNLogic models the probability of $\boldsymbol{a}$ conditioned on $\boldsymbol{q}$ and the existing knowledge graph $\mathcal{G}$, i.e. $p(\boldsymbol{a}|\mathcal{G},\boldsymbol{q})$, where a set of logic rules $\boldsymbol{z}$ is treated as a latent variable. The rule generator defines a prior distribution $p_\theta(\boldsymbol{z}|\boldsymbol{q})$ and the reasoning predictor provides the likelihood function $p_w(\boldsymbol{a}|G, \boldsymbol{q}, \boldsymbol{z})$. See the following figure for illustration.

<img src="./figures/workflow.png" alt="workflow" style="zoom:25%;" />

To optimize the reasoning predictor and the rule generator, we propose an **EM-based algorithm**. At each iteration, the algorithm starts with generating a set of logic rules, which are fed into the reasoning predictor and we further update the reasoning predictor based on the training queries and answers.

<img src="./figures/pre-step.png" alt="pre-step" style="zoom:25%;" />

Then in the E-step, a set of high-quality logic rules are selected from all the generated logic rules according to their posterior probabilities.

<img src="./figures/e-step.png" alt="e-step" style="zoom:25%;" />

Finally in the M-step, the rule generator is updated to be consistent with the high-logic logic rules identified in the E-step.

<img src="./figures/m-step.png" alt="m-step" style="zoom:25%;" />

## Data
We provide four datasets for knowledge graph reasoning, and these datasets are FB15k-237, WN18RR, Kinship, UMLS. For FB15k-237 and WN18RR, there are standard splits for the training/validation/test sets. For Kinship and UMLS, we split the training/validation/test sets by ourselves, and the details are available in our paper.

## Usage
To run RNNLogic, please first go to the directory *pyrnnlogiclib* and run the following command:
```
python setup.py install
```
After that, you can go back to the main directory and use the following commands to do knowledge graph reasoning:
```
python run.py --data_path ./data/wn18rr --num_generated_rules 200 --num_rules_for_test 200 --num_important_rules 0 --prior_weight 0.01 --cuda
python run.py --data_path ./data/kinship --num_generated_rules 2000 --num_rules_for_test 200 --num_important_rules 0 --prior_weight 0.01 --cuda
python run.py --data_path ./data/umls --num_generated_rules 2000 --num_rules_for_test 100 --num_important_rules 0 --prior_weight 0.01 --cuda
```

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