# RNNLogic

This is an implementation of the [RNNLogic](https://openreview.net/forum?id=tGZu6DlbreV) model for knowledge graph reasoning. 

Note that for now this repository only provides the codes of *RNNLogic w/o emb*, and the complete version will be released soon together with a graph reasoning package implemented by our research group **MilaGraph**. Stay tuned!

Also, the current codes are lack of enough comments and **we are keep updating the repository**.

## Data
We provide four datasets for knowledge graph reasoning, and these datasets are FB15k-237, WN18RR, Kinship, UMLS. For FB15k-237 and WN18RR, there are standard splits for the training/validation/test sets. For Kinship and UMLS, we split the training/validation/test sets by ourselves, and the details are available in our paper.

## Usage
To run RNNLogic, please first go to the directory *pyrnnlogiclib* and run the following command:
```
python setup.py install
```
After that, you can go back to the main directory and use the following commands to do knowledge graph reasoning:
```
python run.py --data_path ./data/wn18rr --num_generated_rules 200 --num_rules_for_test 200 --num_important_rules 0 --generator_tune_epochs 100 --generator_tune_learning_rate 1e-5 --prior_weight 0.01 --cuda
python run.py --data_path ./data/kinship --num_generated_rules 2000 --num_rules_for_test 200 --num_important_rules 0 --generator_tune_epochs 100 --generator_tune_learning_rate 1e-5 --prior_weight 0.01 --cuda
python run.py --data_path ./data/umls --num_generated_rules 2000 --num_rules_for_test 200 --num_important_rules 0 --generator_tune_epochs 100 --generator_tune_learning_rate 1e-5 --prior_weight 0.01 --cuda
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