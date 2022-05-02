# RNNLogic+

In this folder, we provide the refactored codes of RNNLogic+, an improved version of RNNLogic as introduced in the Section 3.4 of the paper.

The idea of RNNLogic+ is to first learn useful logic rules by running *RNNLogic w/o emb*, and then use these logic rules to train a more powerful predictor for reasoning. In this way, RNNLogic+ achieves close results to *RNNLogic with emb*, even though no knowledge graph embeddings are used.

To run RNNLogic+, you might follow the following steps.

## Step 1: Mine logic rules

In the first step, we mine some low-quality logic rules, which are used to pre-train the rule generator in RNNLogic+ to speed up training.

To do that, go to the folder `miner`, and compile the codes by running the following command:

`g++ -O3 rnnlogic.h rnnlogic.cpp main.cpp -o rnnlogic -lpthread`

Afterwards, run the following command to mine logic rules:

`./rnnlogic -data-path ../data/FB15k-237 -max-length 3 -threads 40 -lr 0.01 -wd 0.0005 -temp 100 -iterations 1 -top-n 0 -top-k 0 -top-n-out 0 -output-file mined_rules.txt`

The codes run on CPUs. Thus it is better to use a server with many CPUs and use more threads by adjusing the option `-thread`. The program will output a file called `mined_rules.txt`, and you can move the file to your dataset folder.

**In `data/FB15k-237` and `data/wn18rr`, we have provided these mined rules, so you can skip this step.**

## Step 2: Run RNNLogic+

Next, we are ready to run RNNLogic. To do that, please first edit the config file in the folder `config`, and then go to folder `src`.

If you would like to use single-GPU training, please edit line 39 and line 60, and further run:

`python run_rnnlogic.py --config ../config/FB15k-237.yaml` 
`python run_rnnlogic.py --config ../config/wn18rr.yaml` 

If you would like to use multi-GPU training, please run:

`python -m torch.distributed.launch --nproc_per_node=4 run_rnnlogic.py --config ../config/FB15k-237.yaml`
`python -m torch.distributed.launch --nproc_per_node=4 run_rnnlogic.py --config ../config/wn18rr.yaml`

## Results and Discussion

Using the defaul configuration files, we are able to achieve the following results without using knowledge graph embeddings:

**FB15k-237:**

```
Hit1 : 0.242949
Hit3 : 0.358812
Hit10: 0.494145
MR   : 384.201315
MRR  : 0.327182
```

**WN18RR:**

```
Hit1 : 0.439614
Hit3 : 0.483718
Hit10: 0.537939
MR   : 6377.744942
MRR  : 0.471933
```

**Discussion:**

Note that for efficiency consideration, the default configurations are quite conservative, and it is easy to further improve the results.

For example:

- Current configuration files only consider logic rules which are not longer than 3. You might consider longer logic rules for better reasoning results.
- Current configuration files specify the training iterations to 5. You might increase the value for better results.
- Current configuration files specify the hidden dimension in the reasoning predictor to 16. You might also increase the value for better results.
