#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <algorithm>

#define MAX_STRING 1000
#define MAX_THREADS 100
#define MAX_LENGTH 100

double sigmoid(double x);
double abs_val(double x);

struct ArgStruct
{
    void *ptr;
    int id;
    
    ArgStruct(void *_ptr, int _id);
};

struct Triplet
{
    int h, t, r;
    
    friend bool operator < (Triplet u, Triplet v);
    friend bool operator == (Triplet u, Triplet v);
};

struct RankListEntry
{
    int id;
    double val;
    
    friend bool operator < (RankListEntry u, RankListEntry v);
};

struct Parameter
{
    double data, m, v, t;
    
    Parameter();

    void clear();
    void update(double grad, double learning_rate, double weight_decay=0);
};

struct Rule
{
    std::vector<int> r_body;
    int r_head;
    int type;
    double H, cn, prior;
    Parameter wt;
    
    Rule();
    ~Rule();

    void clear();
    friend bool operator < (Rule u, Rule v);
    friend bool operator == (Rule u, Rule v);
};

struct Result
{
    double h1, h3, h10, mr, mrr;

    Result();
    Result(double mr_, double mrr_, double h1_, double h3_, double h10_);
};

class KnowledgeGraph
{
protected:
    int entity_size, relation_size, train_triplet_size, valid_triplet_size, test_triplet_size, all_triplet_size;
    std::map<std::string, int> ent2id, rel2id;
    std::map<int, std::string> id2ent, id2rel;
    std::vector<Triplet> train_triplets, valid_triplets, test_triplets;
    std::vector<std::pair<int, int> > *e2rn;
    std::set<Triplet> set_train_triplets, set_all_triplets;

public:
    friend class RuleMiner;
    friend class ReasoningPredictor;
    friend class RuleGenerator;
    
    KnowledgeGraph();
    ~KnowledgeGraph();

    int get_entity_size();
    int get_relation_size();
    
    void read_data(char *data_path);
    bool check_observed(Triplet triplet);
    bool check_true(Triplet triplet);
    void rule_search(int r, int e, int goal, int *path, int depth, int max_depth, std::set<Rule> &rule_set);
    void rule_destination(int e, Rule rule, std::vector<int> &dests);
};

class RuleMiner
{
protected:
    KnowledgeGraph *p_kg;
    int num_threads, max_length;
    double portion;
    long long total_count;
    std::vector<Rule> *rel2rules;
    std::set<Rule> *rel2ruleset;
    sem_t mutex;

public:
    RuleMiner();
    ~RuleMiner();
    
    void init_knowledge_graph(KnowledgeGraph *_p_kg);
    void clear();
    std::vector<Rule> *get_logic_rules();
    int get_relation_size();
    
    void search_thread(int thread);
    static void *search_thread_caller(void *arg);
    void search(int _max_length, double _portion, int _num_threads);

    void save(char *file_name);
    void load(char *file_name);
};

class ReasoningPredictor
{
protected:
    KnowledgeGraph *p_kg;
    std::vector<Rule> *rel2rules;
    int num_threads, top_k;
    double temperature, learning_rate, weight_decay;
    double portion;
    double prior_weight, H_temperature;
    bool test;
    long long total_count;
    double total_loss;
    std::vector< std::pair<int, int> > ranks;
    sem_t mutex;

public:
    ReasoningPredictor();
    ~ReasoningPredictor();
    
    void init_knowledge_graph(KnowledgeGraph *_p_kg);
    void set_logic_rules(std::vector<Rule> * _rel2rules);
    std::vector<Rule> *get_logic_rules();
    int get_relation_size();
    
    void learn_thread(int thread);
    static void *learn_thread_caller(void *arg);
    void learn(double _learning_rate, double _weight_decay, double _temperature, double _portion, int _num_threads);
    
    void H_score_thread(int thread);
    static void *H_score_thread_caller(void *arg);
    void H_score(int _top_k, double _H_temperature, double _prior_weight, double _portion, int _num_threads);
    
    void evaluate_thread(int thread);
    static void *evaluate_thread_caller(void *arg);
    Result evaluate(bool _test, int _num_threads);
};

class RuleGenerator
{
protected:
    KnowledgeGraph *p_kg;
    std::vector<Rule> *rel2rules, *rel2pool;
    std::vector<int> *mapping;
public:
    RuleGenerator();
    ~RuleGenerator();
    
    void init_knowledge_graph(KnowledgeGraph *_p_kg);
    void set_logic_rules(std::vector<Rule> * _rel2rules);
    std::vector<Rule> *get_logic_rules();
    
    void set_pool(std::vector<Rule> * _rel2rules);
    void sample_from_pool(int _number, double _temperature=1);
    void random_from_pool(int _number);
    void best_from_pool(int _number);
    void update(std::vector<Rule> * _rel2rules);
};
