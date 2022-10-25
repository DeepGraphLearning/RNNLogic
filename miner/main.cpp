#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include "rnnlogic.h"

char data_path[MAX_STRING], output_file[MAX_STRING];
int max_length = 2, num_threads = 1, iterations = 10, top_k = 10, top_n = 100, top_n_out = 100;
double total_loss = 0, learning_rate = 0.01, weight_decay = 0.0, temperature = 100.0;
double miner_portion = 1.0, predictor_portion = 1.0;

KnowledgeGraph KG;
RuleMiner RM;
ReasoningPredictor RP;
RuleGenerator RG;
Result result;

void train()
{
    printf("%lf %lf\n", miner_portion, predictor_portion);

    KG.read_data(data_path);
    
    RM.init_knowledge_graph(&KG);
    RM.search(max_length, miner_portion, num_threads);
    
    RP.init_knowledge_graph(&KG);
    RG.init_knowledge_graph(&KG);
    RG.set_pool(RM.get_logic_rules());

    for (int k = 0; k != iterations; k++)
    {
        RG.random_from_pool(top_n);
        RP.set_logic_rules(RG.get_logic_rules());
        RP.learn(learning_rate, weight_decay, temperature, false, predictor_portion, num_threads);
        RP.H_score(top_k, 1, 0, predictor_portion, num_threads);
        RG.update(RP.get_logic_rules());
    }
    RG.out_rules(output_file, top_n_out);
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a]))
    {
        if (a == argc - 1)
        {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    if (argc == 1)
    {
        return 0;
    }
    data_path[0] = 0;
    if ((i = ArgPos((char *)"-data-path", argc, argv)) > 0) strcpy(data_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-output-file", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-max-length", argc, argv)) > 0) max_length = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iterations", argc, argv)) > 0) iterations = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) learning_rate = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-wd", argc, argv)) > 0) weight_decay = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-temp", argc, argv)) > 0) temperature = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-top-k", argc, argv)) > 0) top_k = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-top-n", argc, argv)) > 0) top_n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-top-n-out", argc, argv)) > 0) top_n_out = atoi(argv[i + 1]);
    
    if (top_n == 0) top_n = 2000000000;
    if (top_n_out == 0) top_n_out = 2000000000;
    
    train();
    return 0;
}
