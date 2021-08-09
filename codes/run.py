import sys
import os

from merge_results import calc_result
import rule_sample
from knowledge_graph_utils import *
from model_rnnlogic import RNNLogic


DATA_DIR             = sys.argv[1]
OUTPUT_DIR           = sys.argv[2]
write_log_to_console = True         if len(sys.argv) <= 3 else eval(sys.argv[3])
start                = 0            if len(sys.argv) <= 4 else int(sys.argv[4])
hop                  = 1            if len(sys.argv) <= 5 else int(sys.argv[5])
RotatE               = 'RotatE_500' if len(sys.argv) <= 6 else sys.argv[6]
hyperparams          =  {
    # See model_rnnlogic.py for default values
    'rotate_pretrained': f"{DATA_DIR}/{RotatE}",
    
    # other hyperparameters here
    
}
if len(sys.argv) > 7:
    hyperparams = dict(sys.argv[7])


'''
Example:
The script will train a separate model for each relation in range(start, total number of relations, hop).

DATA_DIR             = "../data/kinship"
OUTPUT_DIR           = "./workspace"
write_log_to_console = True
start                = 0
hop                  = 1
RotatE               = 'RotatE_500'
'''

old_print = print
log_filename = f"{OUTPUT_DIR}/train_log.txt"
log_file = open(log_filename, 'a')

def new_print(*args, **kwargs):
    if write_log_to_console:
        old_print(*args, **kwargs, flush=True)
    old_print(*args, **kwargs, file=log_file, flush=True)

print = new_print

# Step 0: Install dependencies
os.chdir('./cppext')
os.popen('python setup.py install')
os.chdir('..')

# Step 1: Load dataset
dataset = load_dataset(f"{DATA_DIR}")

# Step 2: Generate rules
# Note: This step only needs to do once.
rule_sample.use_graph(dataset_graph(dataset, 'train'))
for r in range(start, dataset['R'], hop):
    # Usage: rule_sample.sample(relation, dict: rule_len -> num_per_sample, num_samples, ...)
    rules = rule_sample.sample(r, {1: 1, 2: 10, 3: 10, 4: 10}, 1000, num_threads=12, samples_per_print=100)
    rule_sample.save(rules, f"{DATA_DIR}/Rules/rules_{r}.txt")


# Step 3: Create RNNLogic Model
model = RNNLogic(dataset, hyperparams, print=print)

for name, param in model.named_parameters():
    model.print(f"Model Parameter: {name} ({param.type()}:{param.size()})")

# Step 4: Train and output test results.
for r in range(start, dataset['R'], hop):
    model.train_model(r,
                      rule_file=f"{DATA_DIR}/Rules/rules_{r}.txt",
                      model_file=f"{OUTPUT_DIR}/model_{r}.pth")

# Step 5: Merge results, if (start, hop) == (0, 1)
if (start, hop) == (0, 1):
    calc_result(log_filename)
