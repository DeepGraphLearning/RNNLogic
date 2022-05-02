#include "rnnlogic.h"

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double abs_val(double x)
{
    if (x < 0) return -x;
    else return x;
}

/*****************************
    ArgStruct
*****************************/

ArgStruct::ArgStruct(void *_ptr, int _id)
{
    ptr = _ptr;
    id = _id;
}

/*****************************
    Triplet
*****************************/
    
bool operator < (Triplet u, Triplet v)
{
    if (u.r == v.r)
    {
        if (u.h == v.h) return u.t < v.t;
        return u.h < v.h;
    }
    return u.r < v.r;
}
    
bool operator == (Triplet u, Triplet v)
{
    if (u.h == v.h && u.t == v.t && u.r == v.r) return true;
    return false;
}

/*****************************
    RankListEntry
*****************************/
    
bool operator < (RankListEntry u, RankListEntry v)
{
    return u.val > v.val;
}

/*****************************
    Parameter
*****************************/

Parameter::Parameter()
{
    data = 0; m = 0; v = 0; t = 0;
}
    
void Parameter::clear()
{
    data = 0; m = 0; v = 0; t = 0;
}
    
void Parameter::update(double grad, double learning_rate, double weight_decay)
{
    double g = grad - weight_decay * data;

    t += 1;
    m = 0.9 * m + 0.1 * g;
    v = 0.999 * v + 0.001 * g * g;

    double bias1 = 1 - exp(log(0.9) * t);
    double bias2 = 1 - exp(log(0.999) * t);

    double mt = m / bias1;
    double vt = sqrt(v) / sqrt(bias2) + 0.00000001;

    data += learning_rate * mt / vt;
}

/*****************************
    Rule
*****************************/
    
Rule::Rule()
{
    r_body.clear(); r_head = -1;
    type = -1;
    H = 0;
    cn = 0;
    prior = 0;
    wt.clear();
}

Rule::~Rule()
{
    r_body.clear(); r_head = -1;
    type = -1;
    H = 0;
    cn = 0;
    prior = 0;
    wt.clear();
}

void Rule::clear()
{
    r_body.clear(); r_head = -1;
    type = -1;
    H = 0;
    cn = 0;
    prior = 0;
    wt.clear();
}

bool operator < (Rule u, Rule v)
{
    if (u.type == v.type)
    {
        if (u.r_head == v.r_head)
        {
            for (int k = 0; k != u.type; k++)
            {
                if (u.r_body[k] != v.r_body[k])
                return u.r_body[k] < v.r_body[k];
            }
        }
        return u.r_head < v.r_head;
    }
    return u.type < v.type;
}

bool operator == (Rule u, Rule v)
{
    if (u.r_body == v.r_body && u.r_head == v.r_head && u.type == v.type) return true;
    return false;
}

/*****************************
    Result
*****************************/

Result::Result()
{
    h1 = 0; h3 = 0; h10 = 0; mr = 0; mrr = 0;
}

Result::Result(double mr_, double mrr_, double h1_, double h3_, double h10_)
{
    h1 = h1_; h3 = h3_; h10 = h10_; mr = mr_; mrr = mrr_;
}

/*****************************
    KnowledgeGraph
*****************************/
    
KnowledgeGraph::KnowledgeGraph()
{
    entity_size = 0; relation_size = 0;
    train_triplet_size = 0; valid_triplet_size = 0; test_triplet_size = 0;
    all_triplet_size = 0;
    
    ent2id.clear(); rel2id.clear();
    id2ent.clear(); id2rel.clear();
    train_triplets.clear(); valid_triplets.clear(); test_triplets.clear();
    set_train_triplets.clear(); set_all_triplets.clear();
    e2r2n = NULL;
}

KnowledgeGraph::~KnowledgeGraph()
{
    ent2id.clear(); rel2id.clear();
    id2ent.clear(); id2rel.clear();
    train_triplets.clear(); valid_triplets.clear(); test_triplets.clear();
    set_train_triplets.clear(); set_all_triplets.clear();
    for (int k = 0; k != entity_size; k++)
    {
        for (int r = 0; r != relation_size; r++)
            e2r2n[k][r].clear();
        delete [] e2r2n[k];
    }
    delete [] e2r2n;
}

int KnowledgeGraph::get_entity_size()
{
    return entity_size;
}

int KnowledgeGraph::get_relation_size()
{
    return relation_size;
}

int KnowledgeGraph::get_train_size()
{
    return train_triplet_size;
}

int KnowledgeGraph::get_valid_size()
{
    return valid_triplet_size;
}

int KnowledgeGraph::get_test_size()
{
    return test_triplet_size;
}

void KnowledgeGraph::read_data(char *data_path)
{
    char s_head[MAX_STRING], s_tail[MAX_STRING], s_ent[MAX_STRING], s_rel[MAX_STRING], s_file[MAX_STRING];
    int h, t, r, id;
    Triplet triplet;
    std::map<std::string, int>::iterator iter;
    FILE *fi;
    
    strcpy(s_file, data_path);
    strcat(s_file, "/entities.dict");
    fi = fopen(s_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of entities not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%d %s", &id, s_ent) != 2) break;
        
        ent2id[s_ent] = id;
        id2ent[id] = s_ent;
        entity_size += 1;
    }
    fclose(fi);
    
    strcpy(s_file, data_path);
    strcat(s_file, "/relations.dict");
    fi = fopen(s_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of relations not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%d %s", &id, s_rel) != 2) break;
        
        rel2id[s_rel] = id;
        id2rel[id] = s_rel;
        relation_size += 1;
    }
    fclose(fi);
    
    strcpy(s_file, data_path);
    strcat(s_file, "/train.txt");
    fi = fopen(s_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of train triplets not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s", s_head, s_rel, s_tail) != 3) break;
        if (ent2id.count(s_head) == 0 || ent2id.count(s_tail) == 0 || rel2id.count(s_rel) == 0) continue;
        
        h = ent2id[s_head]; t = ent2id[s_tail]; r = rel2id[s_rel];
        triplet.h = h; triplet.t = t; triplet.r = r;
        train_triplets.push_back(triplet);
        set_train_triplets.insert(triplet);
        set_all_triplets.insert(triplet);
    }
    fclose(fi);
    
    train_triplet_size = int(train_triplets.size());
    e2r2n = new std::vector<int> * [entity_size];
    for (int k = 0; k != entity_size; k++) e2r2n[k] = new std::vector<int> [relation_size];
    for (int k = 0; k != train_triplet_size; k++)
    {
        h = train_triplets[k].h; r = train_triplets[k].r; t = train_triplets[k].t;
        e2r2n[h][r].push_back(t);
    }

    strcpy(s_file, data_path);
    strcat(s_file, "/valid.txt");
    fi = fopen(s_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of test triplets not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s", s_head, s_rel, s_tail) != 3) break;
        if (ent2id.count(s_head) == 0 || ent2id.count(s_tail) == 0 || rel2id.count(s_rel) == 0) continue;

        h = ent2id[s_head]; t = ent2id[s_tail]; r = rel2id[s_rel];
        triplet.h = h; triplet.t = t; triplet.r = r;
        valid_triplets.push_back(triplet);
        set_all_triplets.insert(triplet);
    }
    fclose(fi);
    valid_triplet_size = int(valid_triplets.size());
    
    strcpy(s_file, data_path);
    strcat(s_file, "/test.txt");
    fi = fopen(s_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of test triplets not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s", s_head, s_rel, s_tail) != 3) break;
        if (ent2id.count(s_head) == 0 || ent2id.count(s_tail) == 0 || rel2id.count(s_rel) == 0) continue;

        h = ent2id[s_head]; t = ent2id[s_tail]; r = rel2id[s_rel];
        triplet.h = h; triplet.t = t; triplet.r = r;
        test_triplets.push_back(triplet);
        set_all_triplets.insert(triplet);
    }
    fclose(fi);
    test_triplet_size = int(test_triplets.size());

    all_triplet_size = int(set_all_triplets.size());
    
    printf("#Entities: %d          \n", entity_size);
    printf("#Relations: %d          \n", relation_size);
    printf("#Train triplets: %d          \n", train_triplet_size);
    printf("#Valid triplets: %d          \n", valid_triplet_size);
    printf("#Test triplets: %d          \n", test_triplet_size);
    printf("#All triplets: %d          \n", all_triplet_size);
}

bool KnowledgeGraph::check_observed(Triplet triplet)
{
    if (set_train_triplets.count(triplet) != 0) return true;
    else return false;
}

bool KnowledgeGraph::check_true(Triplet triplet)
{
    if (set_all_triplets.count(triplet) != 0) return true;
    else return false;
}

void KnowledgeGraph::rule_search(int r, int e, int goal, int *path, int depth, int max_depth, std::set<Rule> *rule_set, Triplet removed_triplet)
{
    if (e == goal)
    {
        Rule rule;
        rule.type = depth;
        rule.r_head = r;
        rule.r_body.clear();
        for (int k = 0; k != depth; k++)
        {
            rule.r_body.push_back(path[k]);
        }
        rule_set->insert(rule);
        return;
    }
    if (depth == max_depth)
    {
        return;
    }

    int cur_r, cur_n, len;
    for (cur_r = 0; cur_r != relation_size; cur_r++)
    {
        len = int(e2r2n[e][cur_r].size());
        for (int k = 0; k != len; k++)
        {
            cur_n = e2r2n[e][cur_r][k];
            if (e == removed_triplet.h && cur_r == removed_triplet.r && cur_n == removed_triplet.t) continue;
            path[depth] = cur_r;
            rule_search(r, cur_n, goal, path, depth+1, max_depth, rule_set, removed_triplet);
        }
    }
}

/*
void KnowledgeGraph::rule_destination(int e, Rule rule, std::vector<int> &dests, Triplet removed_triplet)
{
    std::queue< std::pair<int, int> > queue;
    queue.push(std::make_pair(e, 0));
    int current_e, current_d, current_r, next_e;
    while (!queue.empty())
    {
        std::pair<int, int> pair = queue.front();
        current_e = pair.first;
        current_d = pair.second;
        queue.pop();
        if (current_d == int(rule.r_body.size()))
        {
            dests.push_back(current_e);
            continue;
        }
        current_r = rule.r_body[current_d];
        for (int k = 0; k != int(e2r2n[current_e][current_r].size()); k++)
        {
            next_e = e2r2n[current_e][current_r][k];
            if (current_e == removed_triplet.h && current_r == removed_triplet.r && next_e == removed_triplet.t) continue;
            queue.push(std::make_pair(next_e, current_d + 1));
        }
    }
}
*/

void KnowledgeGraph::rule_destination(int e, Rule rule, std::map<int, int> *dest2count, Triplet removed_triplet)
{
    std::map<int, int> m[rule.r_body.size() + 1];
    std::map<int, int> *pt;
    std::map<int, int>::iterator iter;
    int current_e, current_r, current_n, next_e;

    (*dest2count).clear();
    m[0][e] = 1;
    for (int k = 0; k != int(rule.r_body.size()); k++)
    {
        if (k == int(rule.r_body.size()) - 1) pt = dest2count;
        else pt = &(m[k + 1]);

        for (iter = m[k].begin(); iter != m[k].end(); iter++)
        {
            current_e = iter->first;
            current_n = iter->second;
            current_r = rule.r_body[k];

            for (int i = 0; i != int(e2r2n[current_e][current_r].size()); i++)
            {
                next_e = e2r2n[current_e][current_r][i];
                if (current_e == removed_triplet.h && current_r == removed_triplet.r && next_e == removed_triplet.t) continue;
                
                if ((*pt).count(next_e) == 0) (*pt)[next_e] = 0;
                (*pt)[next_e] += current_n;
            }
        }
    }
}

/*****************************
    RuleMiner
*****************************/

RuleMiner::RuleMiner()
{
    num_threads = 4; max_length = 3;
    portion = 1;
    total_count = 0;
    rel2ruleset = NULL; rel2rules = NULL;
    sem_init(&mutex, 0, 1);
    p_kg = NULL;
}

RuleMiner::~RuleMiner()
{
    total_count = 0;
    if (rel2ruleset != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2ruleset[r].clear();
        delete [] rel2ruleset;
        rel2ruleset = NULL;
    }
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
        rel2rules = NULL;
    }
    sem_init(&mutex, 0, 1);
    p_kg = NULL;
}

void RuleMiner::init_knowledge_graph(KnowledgeGraph *_p_kg)
{
    p_kg = _p_kg;
    rel2rules = new std::vector<Rule> [p_kg->relation_size];
    rel2ruleset = new std::set<Rule> [p_kg->relation_size];
}

void RuleMiner::clear()
{
    total_count = 0;
    for (int k = 0; k != p_kg->relation_size; k++)
    {
        rel2rules[k].clear();
        rel2ruleset[k].clear();
    }
    sem_init(&mutex, 0, 1);
}

std::vector<Rule> *RuleMiner::get_logic_rules()
{
    return rel2rules;
}

int RuleMiner::get_relation_size()
{
    return p_kg->relation_size;
}
    
void RuleMiner::search_thread(int thread)
{
    int triplet_size = p_kg->train_triplet_size;
    int bg = int(triplet_size / num_threads) * thread;
    int ed = bg + int(triplet_size / num_threads * portion);
    if (thread == num_threads - 1 && portion == 1) ed = triplet_size;
    
    std::set<Rule>::iterator iter;
    std::set<Rule> rule_set;
    std::vector<int> dests;
    Rule rule;
    int path[MAX_LENGTH], h, r, t;
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Rule Discovery | Progress: %.3lf%%          %c", (double)total_count / (double)(triplet_size * portion + 1) * 100, 13);
            fflush(stdout);
        }
        
        h = p_kg->train_triplets[T].h;
        r = p_kg->train_triplets[T].r;
        t = p_kg->train_triplets[T].t;
        
        rule_set.clear();
        p_kg->rule_search(r, h, t, path, 0, max_length, &rule_set, p_kg->train_triplets[T]);
    
        for (iter = rule_set.begin(); iter != rule_set.end(); iter++)
        {
            if (iter->type == 1 && iter->r_body[0] == r)
            {
                rule_set.erase(iter);
                break;
            }
        }
        
        for (iter = rule_set.begin(); iter != rule_set.end(); iter++)
        {
            rule = *iter;
            sem_wait(&mutex);
            rel2ruleset[r].insert(rule);
            sem_post(&mutex);
        }
    }
    dests.clear();
    rule_set.clear();
    pthread_exit(NULL);
}

void *RuleMiner::search_thread_caller(void *arg)
{
    RuleMiner *ptr = (RuleMiner *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->search_thread(thread);
    pthread_exit(NULL);
}

void RuleMiner::search(int _max_length, double _portion, int _num_threads)
{
    max_length = _max_length;
    portion = _portion;
    num_threads = _num_threads;

    std::random_shuffle((p_kg->train_triplets).begin(), (p_kg->train_triplets).end());
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, RuleMiner::search_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    printf("Rule Discovery | DONE!                              \n");
    free(pt);

    int rel;
    Rule rule;
    std::set<Rule>::iterator iter;
    for (rel = 0; rel != p_kg->relation_size; rel++)
    {
        for (iter = rel2ruleset[rel].begin(); iter != rel2ruleset[rel].end(); iter++)
        {
            rule = *iter;
            rel2rules[rel].push_back(rule);
        }
    }
}

void RuleMiner::save(char *file_name)
{
    FILE *fo = fopen(file_name, "wb");
    int cn = 0;
    Rule rule;
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        for (int k = 0; k != int(rel2rules[r].size()); k++)
        {
            rule = rel2rules[r][k];
            fprintf(fo, "%d %d", rule.type, rule.r_head);
            for (int k = 0; k != int(rule.r_body.size()); k++)
                fprintf(fo, " %d", rule.r_body[k]);
            fprintf(fo, " %lf %lf %lf\n", rule.H, rule.wt.data, rule.prior);

            cn += 1;
        }
    }
    fclose(fo);

    printf("#Logic rules saved: %d\n", cn);
}

void RuleMiner::load(char *file_name)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
        
    rel2rules = new std::vector<Rule> [p_kg->relation_size];

    FILE *fi = fopen(file_name, "rb");
    int type, r_head, r_body, cn = 0;
    double H, wt, prior;
    Rule rule;
    while (1)
    {
        if (fscanf(fi, "%d %d", &type, &r_head) != 2) break;

        rule.clear();
        rule.type = type;
        rule.r_head = r_head;
        for (int k = 0; k != type; k++)
        {
            if (fscanf(fi, "%d", &r_body) != 1)
            {
                printf("ERROR: format error in rule files!\n");
                exit(1);
            }
            rule.r_body.push_back(r_body);
        }
        if (fscanf(fi, "%lf %lf %lf", &H, &wt, &prior) != 3)
        {
            printf("ERROR: format error in rule files!\n");
            exit(1);
        }
        rule.H = H;
        rule.wt.data = wt;
        rule.prior = prior;

        rel2rules[r_head].push_back(rule);
        cn += 1;
    }
    fclose(fi);

    printf("#Logic rules loaded: %d\n", cn);
}

/*****************************
    ReasoningPredictor
*****************************/

ReasoningPredictor::ReasoningPredictor()
{
    num_threads = 4; top_k = 100;
    temperature = 100; learning_rate = 0.01; weight_decay = 0.0005;
    portion = 1.0;
    prior_weight = 0; H_temperature = 1;
    total_count = 0; total_loss = 0;
    rel2rules = NULL;
    test = true; fast = true;
    ranks.clear();
    sem_init(&mutex, 0, 1);
    p_kg = NULL;
}

ReasoningPredictor::~ReasoningPredictor()
{
    total_count = 0; total_loss = 0;
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
        rel2rules = NULL;
    }
    ranks.clear();
    sem_init(&mutex, 0, 1);
    p_kg = NULL;
}

void ReasoningPredictor::init_knowledge_graph(KnowledgeGraph *_p_kg)
{
    p_kg = _p_kg;
}

void ReasoningPredictor::set_logic_rules(std::vector<Rule> * _rel2rules)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
    
    rel2rules = new std::vector<Rule> [p_kg->relation_size];
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        rel2rules[r] = _rel2rules[r];
        for (int k = 0; k != int(rel2rules[r].size()); k++)
        {
            rel2rules[r][k].wt.clear();
            rel2rules[r][k].H = 0;
        }
    }
}

std::vector<Rule> *ReasoningPredictor::get_logic_rules()
{
    return rel2rules;
}

int ReasoningPredictor::get_relation_size()
{
    return p_kg->relation_size;
}

void ReasoningPredictor::learn_thread(int thread)
{
    int triplet_size = p_kg->train_triplet_size;
    int bg = int(triplet_size / num_threads) * thread;
    int ed = bg + int(triplet_size / num_threads * portion);
    if (thread == num_threads - 1 && portion == 1) ed = triplet_size;
    
    Triplet triplet;
    int h, r, t, dest, count, index;
    double logit, target, grad;

    std::map<int, double> dest2logit, dest2grad;
    std::map<int, int>::iterator iter_ii;
    std::map<int, double>::iterator iter_id;

    int max_num_rules = 0;
    for (r = 0; r != p_kg->relation_size; r++) if (int(rel2rules[r].size()) > max_num_rules)
        max_num_rules = int(rel2rules[r].size());
    std::map<int, int> *index2dest2count = new std::map<int, int> [max_num_rules];
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Learning Rule Weights | Progress: %.3lf%% | Loss: %.6lf          %c", (double)total_count / (double)(triplet_size * portion + 1) * 100, total_loss / total_count, 13);
            fflush(stdout);
        }
        
        h = p_kg->train_triplets[T].h;
        r = p_kg->train_triplets[T].r;
        t = p_kg->train_triplets[T].t;
        
        dest2logit.clear();
        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            p_kg->rule_destination(h, rel2rules[r][index], &(index2dest2count[index]), p_kg->train_triplets[T]);

            for (iter_ii = index2dest2count[index].begin(); iter_ii != index2dest2count[index].end(); iter_ii++)
            {
                dest = iter_ii->first;
                count = iter_ii->second;

                if (dest2logit.count(dest) == 0) dest2logit[dest] = 0;
                dest2logit[dest] += rel2rules[r][index].wt.data * count / temperature;
            }
        }

        double max_val = -1000000, sum_val = 0;
        for (iter_id = dest2logit.begin(); iter_id != dest2logit.end(); iter_id++)
            max_val = std::max(max_val, iter_id->second);
        for (iter_id = dest2logit.begin(); iter_id != dest2logit.end(); iter_id++)
            sum_val += exp(iter_id->second - max_val);
        for (iter_id = dest2logit.begin(); iter_id != dest2logit.end(); iter_id++)
            dest2logit[iter_id->first] = exp(dest2logit[iter_id->first] - max_val) / sum_val;

        dest2grad.clear();
        for (iter_id = dest2logit.begin(); iter_id != dest2logit.end(); iter_id++)
        {
            dest = iter_id->first;
            logit = iter_id->second;
            
            triplet = p_kg->train_triplets[T];
            triplet.t = dest;
            if (p_kg->check_observed(triplet) == true) target = 1.0;
            else target = 0;
            grad = (target - logit) / temperature;
            dest2grad[dest] = grad;
            
            total_loss += abs_val(target - logit) / dest2logit.size();
        }

        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            for (iter_ii = index2dest2count[index].begin(); iter_ii != index2dest2count[index].end(); iter_ii++)
            {
                dest = iter_ii->first;
                count = iter_ii->second;
                grad = dest2grad[dest];
                if (fast) rel2rules[r][index].wt.update(grad * count, learning_rate, weight_decay);
                else for (int k = 0; k != count; k++) rel2rules[r][index].wt.update(grad, learning_rate, weight_decay);
                
            }
        }
    }
    dest2logit.clear();
    delete [] index2dest2count;
    pthread_exit(NULL);
}

void *ReasoningPredictor::learn_thread_caller(void *arg)
{
    ReasoningPredictor *ptr = (ReasoningPredictor *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->learn_thread(thread);
    pthread_exit(NULL);
}

void ReasoningPredictor::learn(double _learning_rate, double _weight_decay, double _temperature, bool _fast, double _portion, int _num_threads)
{
    learning_rate = _learning_rate;
    weight_decay = _weight_decay;
    temperature = _temperature;
    fast = _fast;
    portion = _portion;
    num_threads = _num_threads;
    
    total_count = 0;
    total_loss = 0;

    std::random_shuffle((p_kg->train_triplets).begin(), (p_kg->train_triplets).end());
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, ReasoningPredictor::learn_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    printf("Learning Rule Weights | DONE! | Loss: %.6lf                             \n", total_loss / total_count);
    free(pt);
}

void ReasoningPredictor::H_score_thread(int thread)
{
    int triplet_size = p_kg->train_triplet_size;
    int bg = int(triplet_size / num_threads) * thread;
    int ed = bg + int(triplet_size / num_threads * portion);
    if (thread == num_threads - 1 && portion == 1) ed = triplet_size;
    
    std::vector<int> dests;
    int h, r, t, dest, count, index;

    std::set<int> destset;
    std::map<int, int>::iterator iter_ii;

    int max_num_rules = 0;
    for (int r = 0; r != p_kg->relation_size; r++) if (int(rel2rules[r].size()) > max_num_rules)
        max_num_rules = int(rel2rules[r].size());
    std::map<int, int> *index2dest2count = new std::map<int, int> [max_num_rules];
    
    RankListEntry *rule2score = new RankListEntry [max_num_rules];

    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Computing H Score | Progress: %.3lf%%          %c", (double)total_count / (double)(triplet_size * portion + 1) * 100, 13);
            fflush(stdout);
        }
        
        h = p_kg->train_triplets[T].h;
        r = p_kg->train_triplets[T].r;
        t = p_kg->train_triplets[T].t;
        
        destset.clear();
        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            p_kg->rule_destination(h, rel2rules[r][index], &(index2dest2count[index]), p_kg->train_triplets[T]);
            
            for (iter_ii = index2dest2count[index].begin(); iter_ii != index2dest2count[index].end(); iter_ii++)
            {
                dest = iter_ii->first;

                if (destset.count(dest) == 0) destset.insert(dest);
            }
        }

        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            rule2score[index].id = index;
            rule2score[index].val = rel2rules[r][index].prior * prior_weight;
            for (iter_ii = index2dest2count[index].begin(); iter_ii != index2dest2count[index].end(); iter_ii++)
            {
                dest = iter_ii->first;
                count = iter_ii->second;

                if (dest == t) rule2score[index].val += rel2rules[r][index].wt.data * count;
                rule2score[index].val -= rel2rules[r][index].wt.data * count / destset.size();
            }
        }

        if (top_k == 0)
        {
            double max_val = -1000000, sum_val = 0;

            for (int k = 0; k != int(rel2rules[r].size()); k++)
                rule2score[k].val /= H_temperature;
            for (int k = 0; k != int(rel2rules[r].size()); k++)
                max_val = std::max(max_val, rule2score[k].val);
            for (int k = 0; k != int(rel2rules[r].size()); k++)
                sum_val += exp(rule2score[k].val - max_val);
            for (int k = 0; k != int(rel2rules[r].size()); k++)
            {
                index = rule2score[k].id;
                rel2rules[r][index].H += exp(rule2score[k].val - max_val) / sum_val / triplet_size;
            }
        }
        else
        {
            std::sort(rule2score, rule2score + int(rel2rules[r].size()));
        
            for (int k = 0; k != int(rel2rules[r].size()); k++)
            {
                if (k == top_k) break;

                index = rule2score[k].id;
                rel2rules[r][index].H += 1.0 / top_k / triplet_size;
            }
        }
    }
    destset.clear();
    delete [] rule2score;
    delete [] index2dest2count;
    pthread_exit(NULL);
}

void *ReasoningPredictor::H_score_thread_caller(void *arg)
{
    ReasoningPredictor *ptr = (ReasoningPredictor *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->H_score_thread(thread);
    pthread_exit(NULL);
}

void ReasoningPredictor::H_score(int _top_k, double _H_temperature, double _prior_weight, double _portion, int _num_threads)
{
    top_k = _top_k;
    H_temperature = _H_temperature;
    prior_weight = _prior_weight;
    portion = _portion;
    num_threads = _num_threads;
    
    total_count = 0;
    total_loss = 0;
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, ReasoningPredictor::H_score_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    printf("Computing H Score | DONE!                              \n");
    free(pt);
}

void ReasoningPredictor::evaluate_thread(int thread)
{
    std::vector<Triplet> *p_triplets;
    if (test) p_triplets = &(p_kg->test_triplets);
    else p_triplets = &(p_kg->valid_triplets);
    
    int triplet_size = int((*p_triplets).size());
    int bg = int(triplet_size / num_threads) * thread;
    int ed = int(triplet_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = triplet_size;
    
    Triplet triplet;
    int h, r, t, dest, count, index, num_g, num_ge;
    double t_val;

    std::map<int, int> dest2count;
    std::map<int, int>::iterator iter_ii;

    RankListEntry *rank_list = new RankListEntry [p_kg->entity_size];
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Evaluation | Progress: %.3lf%%          %c", (double)total_count / (double)(triplet_size + 1) * 100, 13);
            fflush(stdout);
        }
        
        h = (*p_triplets)[T].h;
        r = (*p_triplets)[T].r;
        t = (*p_triplets)[T].t;
        
        for (int k = 0; k != p_kg->entity_size; k++)
        {
            rank_list[k].id = k;
            rank_list[k].val = 0;
        }
        
        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            p_kg->rule_destination(h, rel2rules[r][index], &dest2count, (*p_triplets)[T]);
            
            for (iter_ii = dest2count.begin(); iter_ii != dest2count.end(); iter_ii++)
            {
                dest = iter_ii->first;
                count = iter_ii->second;

                rank_list[dest].val += rel2rules[r][index].wt.data * count;
            }
        }

        t_val = rank_list[t].val;
        
        std::sort(rank_list, rank_list + p_kg->entity_size);

        num_g = 0; num_ge = 0;
        triplet = (*p_triplets)[T];
        for (int k = 0; k != p_kg->entity_size; k++)
        {
            triplet.t = rank_list[k].id;
            if (p_kg->check_true(triplet) == true && rank_list[k].id != t) continue;

            if (rank_list[k].val > t_val) num_g += 1;
            if (rank_list[k].val >= t_val) num_ge += 1;
            if (rank_list[k].val < t_val) break;
        }
        
        sem_wait(&mutex);
        ranks.push_back(std::make_pair(num_g, num_ge));
        sem_post(&mutex);
    }
    dest2count.clear();
    delete [] rank_list;
    pthread_exit(NULL);
}

void *ReasoningPredictor::evaluate_thread_caller(void *arg)
{
    ReasoningPredictor *ptr = (ReasoningPredictor *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->evaluate_thread(thread);
    pthread_exit(NULL);
}

Result ReasoningPredictor::evaluate(bool _test, int _num_threads)
{
    test = _test;
    num_threads = _num_threads;
    
    ranks.clear();
    total_count = 0;
    total_loss = 0;
    sem_init(&mutex, 0, 1);
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, ReasoningPredictor::evaluate_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    if (test == true) printf("Evaluation Test | DONE!                              \n");
    else printf("Evaluation Valid | DONE!                              \n");
    free(pt);

    int num_entities = p_kg->entity_size;
    double *table_mr = (double *)calloc(num_entities + 1, sizeof(double));
    double *table_mrr = (double *)calloc(num_entities + 1, sizeof(double));
    double *table_hit1 = (double *)calloc(num_entities + 1, sizeof(double));
    double *table_hit3 = (double *)calloc(num_entities + 1, sizeof(double));
    double *table_hit10 = (double *)calloc(num_entities + 1, sizeof(double));
    for (int rank = 1; rank <= num_entities; rank++)
    {
        table_mr[rank] = rank;
        table_mrr[rank] = 1.0 / rank;
        if (rank <= 1) table_hit1[rank] = 1;
        if (rank <= 3) table_hit3[rank] = 1;
        if (rank <= 10) table_hit10[rank] = 1;
    }
    for (int rank = 1; rank <= num_entities; rank++)
    {
        table_mr[rank] += table_mr[rank - 1];
        table_mrr[rank] += table_mrr[rank - 1];
        table_hit1[rank] += table_hit1[rank - 1];
        table_hit3[rank] += table_hit3[rank - 1];
        table_hit10[rank] += table_hit10[rank - 1];
    }
    
    double mr = 0, mrr = 0, hit1 = 0, hit3 = 0, hit10 = 0;
    std::vector< std::pair<int, int> >::iterator iter;
    for (iter = ranks.begin(); iter != ranks.end(); iter++)
    {
        int num_g = iter->first;
        int num_ge = iter->second;
        mr += (table_mr[num_ge] - table_mr[num_g]) / (num_ge - num_g);
        mrr += (table_mrr[num_ge] - table_mrr[num_g]) / (num_ge - num_g);
        hit1 += (table_hit1[num_ge] - table_hit1[num_g]) / (num_ge - num_g);
        hit3 += (table_hit3[num_ge] - table_hit3[num_g]) / (num_ge - num_g);
        hit10 += (table_hit10[num_ge] - table_hit10[num_g]) / (num_ge - num_g);
    }

    free(table_mr);
    free(table_mrr);
    free(table_hit1);
    free(table_hit3);
    free(table_hit10);

    mr /= ranks.size();
    mrr /= ranks.size();
    hit1 /= ranks.size();
    hit3 /= ranks.size();
    hit10 /= ranks.size();

    Result result(mr, mrr, hit1, hit3, hit10);
    return result;
}

void ReasoningPredictor::out_train_thread(int thread)
{
    int triplet_size = p_kg->train_triplet_size;
    int bg = int(triplet_size / num_threads) * thread;
    int ed = bg + int(triplet_size / num_threads * portion);
    if (thread == num_threads - 1 && portion == 1) ed = triplet_size;
    
    Triplet triplet;
    int h, r, t, dest, count, index, valid;

    std::map<int, int> dest2count;
    std::map<int, int>::iterator iter_ii;
    std::map<int, std::map<int, int> > dest2index2count;
    std::map<int, std::map<int, int> >::iterator iter_im;
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Generating Data | Progress: %.3lf%%          %c", (double)total_count / (double)(triplet_size * portion + 1) * 100, 13);
            fflush(stdout);
        }
        
        h = p_kg->train_triplets[T].h;
        r = p_kg->train_triplets[T].r;
        t = p_kg->train_triplets[T].t;

        dest2index2count.clear();
        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            p_kg->rule_destination(h, rel2rules[r][index], &dest2count, p_kg->train_triplets[T]);
            
            for (iter_ii = dest2count.begin(); iter_ii != dest2count.end(); iter_ii++)
            {
                dest = iter_ii->first;
                count = iter_ii->second;

                if (dest2index2count.count(dest) == 0) dest2index2count[dest] = std::map<int, int>();
                dest2index2count[dest][index] = count;
            }
        }

        for (iter_im = dest2index2count.begin(); iter_im != dest2index2count.end(); iter_im++)
        {
            dest = iter_im->first;

            valid = 0;
            triplet = p_kg->train_triplets[T];
            triplet.t = dest;
            if (p_kg->check_observed(triplet) == true) valid = 1;

            thread_data[thread].push_back(h);
            thread_data[thread].push_back(r);
            thread_data[thread].push_back(t);
            thread_data[thread].push_back(valid);
            thread_data[thread].push_back(dest);
            thread_data[thread].push_back(int((iter_im->second).size()));

            for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
            {
                thread_data[thread].push_back(iter_ii->first);
            }

            for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
            {
                thread_data[thread].push_back(iter_ii->second);
            }

            thread_split[thread].push_back(int(thread_data[thread].size()));
        }
    }
    dest2count.clear();
    dest2index2count.clear();
    pthread_exit(NULL);
}

void *ReasoningPredictor::out_train_thread_caller(void *arg)
{
    ReasoningPredictor *ptr = (ReasoningPredictor *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->out_train_thread(thread);
    pthread_exit(NULL);
}

void ReasoningPredictor::out_train(std::vector<int> *data, double _portion, int _num_threads)
{
    portion = _portion;
    num_threads = _num_threads;
    
    total_count = 0;
    for (int k = 0; k != num_threads; k++) thread_data[k].clear();
    for (int k = 0; k != num_threads; k++) thread_split[k].clear();

    std::random_shuffle((p_kg->train_triplets).begin(), (p_kg->train_triplets).end());
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, ReasoningPredictor::out_train_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    printf("Generating Training Data | DONE!                             \n");
    free(pt);

    (*data).clear();
    std::vector<int> split;
    split.push_back(0);
    int base = 0;
    for (int k = 0; k != num_threads; k++)
    {
        (*data).insert((*data).end(), thread_data[k].begin(), thread_data[k].end());
        for (int i = 0; i != int(thread_split[k].size()); i++) split.push_back(base + thread_split[k][i]);
        base += int(thread_data[k].size());

        thread_data[k].clear();
        thread_split[k].clear();
    }

    int data_length = int((*data).size());
    for (int k = 0; k != int(split.size()); k++) (*data).push_back(split[k]);
    (*data).push_back(data_length);

    /*if (file_name != NULL)
    {
        FILE *fo = fopen(file_name, "wb");
        Instance instance;
        std::map<int, int>::iterator iter;
        for (int k = 0; k != int(instances.size()); k++)
        {
            instance = instances[k];
            fprintf(fo, "%d %d %d %d\n", instance.h, instance.r, instance.t, int(instance.vec_destrule.size()));
            for (int i = 0; i != int(instance.vec_destrule.size()); i++)
            {
                fprintf(fo, "%d %d %d", instance.vec_destrule[i].valid, instance.vec_destrule[i].dest, int(instance.vec_destrule[i].index2count.size()));
                for (iter = instance.vec_destrule[i].index2count.begin(); iter != instance.vec_destrule[i].index2count.end(); iter++) fprintf(fo, " %d:%d", iter->first, iter->second);
                fprintf(fo, "\n");
            }
        }
        fclose(fo);
    }*/
}

void ReasoningPredictor::out_test_thread(int thread)
{
    std::vector<Triplet> *p_triplets;
    if (test) p_triplets = &(p_kg->test_triplets);
    else p_triplets = &(p_kg->valid_triplets);
    
    int triplet_size = int((*p_triplets).size());
    int bg = int(triplet_size / num_threads) * thread;
    int ed = int(triplet_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = triplet_size;
    
    Triplet triplet;
    int h, r, t, dest, count, index, valid;

    std::map<int, int> dest2count;
    std::map<int, int>::iterator iter_ii;
    std::map<int, std::map<int, int> > dest2index2count;
    std::map<int, std::map<int, int> >::iterator iter_im;
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Generating Data | Progress: %.3lf%%          %c", (double)total_count / (double)(triplet_size + 1) * 100, 13);
            fflush(stdout);
        }
        
        h = (*p_triplets)[T].h;
        r = (*p_triplets)[T].r;
        t = (*p_triplets)[T].t;

        dest2index2count.clear();
        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            p_kg->rule_destination(h, rel2rules[r][index], &dest2count, (*p_triplets)[T]);
            
            for (iter_ii = dest2count.begin(); iter_ii != dest2count.end(); iter_ii++)
            {
                dest = iter_ii->first;
                count = iter_ii->second;

                if (dest2index2count.count(dest) == 0) dest2index2count[dest] = std::map<int, int>();
                dest2index2count[dest][index] = count;
            }
        }

        for (iter_im = dest2index2count.begin(); iter_im != dest2index2count.end(); iter_im++)
        {
            dest = iter_im->first;

            triplet = (*p_triplets)[T];
            triplet.t = dest;
            if (p_kg->check_true(triplet) == true && dest != t) continue;

            valid = 0;
            triplet = (*p_triplets)[T];
            triplet.t = dest;
            if (p_kg->check_true(triplet) == true) valid = 1;

            thread_data[thread].push_back(h);
            thread_data[thread].push_back(r);
            thread_data[thread].push_back(t);
            thread_data[thread].push_back(valid);
            thread_data[thread].push_back(dest);
            thread_data[thread].push_back(int((iter_im->second).size()));

            for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
            {
                thread_data[thread].push_back(iter_ii->first);
            }

            for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
            {
                thread_data[thread].push_back(iter_ii->second);
            }

            thread_split[thread].push_back(int(thread_data[thread].size()));
        }
    }
    dest2count.clear();
    dest2index2count.clear();
    pthread_exit(NULL);
}

void *ReasoningPredictor::out_test_thread_caller(void *arg)
{
    ReasoningPredictor *ptr = (ReasoningPredictor *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->out_test_thread(thread);
    pthread_exit(NULL);
}

void ReasoningPredictor::out_test(std::vector<int> *data, bool _test, int _num_threads)
{
    test = _test;
    num_threads = _num_threads;
    
    total_count = 0;
    for (int k = 0; k != num_threads; k++) thread_data[k].clear();
    for (int k = 0; k != num_threads; k++) thread_split[k].clear();
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, ReasoningPredictor::out_test_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    if (test == true) printf("Generating Test Data | DONE!                             \n");
    else printf("Generating Validation Data | DONE!                             \n");
    free(pt);

    (*data).clear();
    std::vector<int> split;
    split.push_back(0);
    int base = 0;
    for (int k = 0; k != num_threads; k++)
    {
        (*data).insert((*data).end(), thread_data[k].begin(), thread_data[k].end());
        for (int i = 0; i != int(thread_split[k].size()); i++) split.push_back(base + thread_split[k][i]);
        base += int(thread_data[k].size());

        thread_data[k].clear();
        thread_split[k].clear();
    }

    int data_length = int((*data).size());
    for (int k = 0; k != int(split.size()); k++) (*data).push_back(split[k]);
    (*data).push_back(data_length);

    /*if (file_name != NULL)
    {
        FILE *fo = fopen(file_name, "wb");
        Instance instance;
        std::map<int, int>::iterator iter;
        for (int k = 0; k != int(instances.size()); k++)
        {
            instance = instances[k];
            fprintf(fo, "%d %d %d %d\n", instance.h, instance.r, instance.t, int(instance.vec_destrule.size()));
            for (int i = 0; i != int(instance.vec_destrule.size()); i++)
            {
                fprintf(fo, "%d %d %d", instance.vec_destrule[i].valid, instance.vec_destrule[i].dest, int(instance.vec_destrule[i].index2count.size()));
                for (iter = instance.vec_destrule[i].index2count.begin(); iter != instance.vec_destrule[i].index2count.end(); iter++) fprintf(fo, " %d:%d", iter->first, iter->second);
                fprintf(fo, "\n");
            }
        }
        fclose(fo);
    }*/
}

void ReasoningPredictor::in_rules(char *file_name)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
    
    rel2rules = new std::vector<Rule> [p_kg->relation_size];

    FILE *fi = fopen(file_name, "rb");

    int r, r_head, n, id, type, r_body;
    Rule rule;
    while (1)
    {
        if (fscanf(fi, "%d %d", &r, &n) != 2) break;

        for (int k = 0; k != n; k++)
        {
            rule.clear();
            fscanf(fi, "%d %d %d", &id, &r_head, &type);
            rule.r_head = r_head;
            rule.type = type;
            for (int i = 0; i != type; i++)
            {
                fscanf(fi, "%d", &r_body);
                rule.r_body.push_back(r_body);
            }
            rel2rules[r].push_back(rule);
        }
    }
}

void ReasoningPredictor::out_rules(char *file_name)
{
    FILE *fo = fopen(file_name, "wb");
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        fprintf(fo, "%d %d\n", r, int(rel2rules[r].size()));
        for (int k = 0; k != int(rel2rules[r].size()); k++)
        {
            fprintf(fo, "%d %d %d", k, rel2rules[r][k].r_head, rel2rules[r][k].type);
            for (int i = 0; i != int(rel2rules[r][k].r_body.size()); i++) fprintf(fo, " %d", rel2rules[r][k].r_body[i]);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
}

void ReasoningPredictor::out_train_single(int h, int r, int t, std::vector<int> *data)
{
    int dest, count, index, valid;

    std::vector<int> split;
    split.push_back(0);

    std::map<int, int> dest2count;
    std::map<int, int>::iterator iter_ii;
    std::map<int, std::map<int, int> > dest2index2count;
    std::map<int, std::map<int, int> >::iterator iter_im;

    Triplet triplet;
    triplet.h = h; triplet.r = r; triplet.t = t;

    dest2index2count.clear();
    for (index = 0; index != int(rel2rules[r].size()); index++)
    {
        p_kg->rule_destination(h, rel2rules[r][index], &dest2count, triplet);
            
        for (iter_ii = dest2count.begin(); iter_ii != dest2count.end(); iter_ii++)
        {
            dest = iter_ii->first;
            count = iter_ii->second;

            if (dest2index2count.count(dest) == 0) dest2index2count[dest] = std::map<int, int>();
            dest2index2count[dest][index] = count;
        }
    }

    for (iter_im = dest2index2count.begin(); iter_im != dest2index2count.end(); iter_im++)
    {
        dest = iter_im->first;

        valid = 0;
        triplet.h = h;
        triplet.r = r;
        triplet.t = dest;
        if (p_kg->check_observed(triplet) == true) valid = 1;

        (*data).push_back(h);
        (*data).push_back(r);
        (*data).push_back(t);
        (*data).push_back(valid);
        (*data).push_back(dest);
        (*data).push_back(int((iter_im->second).size()));

        for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
        {
            (*data).push_back(iter_ii->first);
        }

        for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
        {
            (*data).push_back(iter_ii->second);
        }

        split.push_back(int((*data).size()));
    }

    int data_length = int((*data).size());
    for (int k = 0; k != int(split.size()); k++) (*data).push_back(split[k]);
    (*data).push_back(data_length);

    dest2count.clear();
    dest2index2count.clear();
}

void ReasoningPredictor::out_test_single(int h, int r, int t, std::vector<int> *data)
{
    int dest, count, index, valid;

    std::vector<int> split;
    split.push_back(0);

    std::map<int, int> dest2count;
    std::map<int, int>::iterator iter_ii;
    std::map<int, std::map<int, int> > dest2index2count;
    std::map<int, std::map<int, int> >::iterator iter_im;

    Triplet triplet;
    triplet.h = h; triplet.r = r; triplet.t = t;

    dest2index2count.clear();
    for (index = 0; index != int(rel2rules[r].size()); index++)
    {
        p_kg->rule_destination(h, rel2rules[r][index], &dest2count, triplet);
            
        for (iter_ii = dest2count.begin(); iter_ii != dest2count.end(); iter_ii++)
        {
            dest = iter_ii->first;
            count = iter_ii->second;

            if (dest2index2count.count(dest) == 0) dest2index2count[dest] = std::map<int, int>();
            dest2index2count[dest][index] = count;
        }
    }

    for (iter_im = dest2index2count.begin(); iter_im != dest2index2count.end(); iter_im++)
    {
        dest = iter_im->first;

        triplet.h = h;
        triplet.r = r;
        triplet.t = dest;
        if (p_kg->check_true(triplet) == true && dest != t) continue;

        valid = 0;
        if (p_kg->check_true(triplet) == true) valid = 1;

        (*data).push_back(h);
        (*data).push_back(r);
        (*data).push_back(t);
        (*data).push_back(valid);
        (*data).push_back(dest);
        (*data).push_back(int((iter_im->second).size()));

        for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
        {
            (*data).push_back(iter_ii->first);
        }

        for (iter_ii = (iter_im->second).begin(); iter_ii != (iter_im->second).end(); iter_ii++)
        {
            (*data).push_back(iter_ii->second);
        }

        split.push_back(int((*data).size()));
    }

    int data_length = int((*data).size());
    for (int k = 0; k != int(split.size()); k++) (*data).push_back(split[k]);
    (*data).push_back(data_length);

    dest2count.clear();
    dest2index2count.clear();
}

void ReasoningPredictor::out_test_count_thread(int thread)
{
    std::vector<Triplet> *p_triplets;
    if (test) p_triplets = &(p_kg->test_triplets);
    else p_triplets = &(p_kg->valid_triplets);
    
    int triplet_size = int((*p_triplets).size());
    int bg = int(triplet_size / num_threads) * thread;
    int ed = int(triplet_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = triplet_size;
    
    Triplet triplet;
    int h, r, t, dest, count, index;

    std::map<int, int> dest2count;
    std::map<int, int>::iterator iter_ii;
    std::set<int> destset;
    std::set<int>::iterator iter_i;
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Generating Count | Progress: %.3lf%%          %c", (double)total_count / (double)(triplet_size + 1) * 100, 13);
            fflush(stdout);
        }
        
        h = (*p_triplets)[T].h;
        r = (*p_triplets)[T].r;
        t = (*p_triplets)[T].t;

        destset.clear();
        for (index = 0; index != int(rel2rules[r].size()); index++)
        {
            p_kg->rule_destination(h, rel2rules[r][index], &dest2count, (*p_triplets)[T]);
            
            for (iter_ii = dest2count.begin(); iter_ii != dest2count.end(); iter_ii++)
            {
                dest = iter_ii->first;

                destset.insert(dest);
            }
        }

        count = 0;
        for (iter_i = destset.begin(); iter_i != destset.end(); iter_i++)
        {
            dest = *iter_i;

            triplet = (*p_triplets)[T];
            triplet.t = dest;
            if (p_kg->check_true(triplet) == true && dest != t) continue;

            count += 1;
        }

        thread_data[thread].push_back(count);
    }
    destset.clear();
    dest2count.clear();
    pthread_exit(NULL);
}

void *ReasoningPredictor::out_test_count_thread_caller(void *arg)
{
    ReasoningPredictor *ptr = (ReasoningPredictor *)(((ArgStruct *)arg)->ptr);
    int thread = ((ArgStruct *)arg)->id;
    ptr->out_test_count_thread(thread);
    pthread_exit(NULL);
}

void ReasoningPredictor::out_test_count(std::vector<int> *data, bool _test, int _num_threads)
{
    test = _test;
    num_threads = _num_threads;
    
    total_count = 0;
    for (int k = 0; k != num_threads; k++) thread_data[k].clear();
    
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int k = 0; k != num_threads; k++) pthread_create(&pt[k], NULL, ReasoningPredictor::out_test_count_thread_caller, new ArgStruct(this, k));
    for (int k = 0; k != num_threads; k++) pthread_join(pt[k], NULL);
    if (test == true) printf("Generating Test Data | DONE!                             \n");
    else printf("Generating Validation Data | DONE!                             \n");
    free(pt);

    (*data).clear();
    for (int k = 0; k != num_threads; k++)
    {
        (*data).insert((*data).end(), thread_data[k].begin(), thread_data[k].end());

        thread_data[k].clear();
    }
}

/*****************************
    RuleGenerator
*****************************/

RuleGenerator::RuleGenerator()
{
    rel2rules = NULL;
    rel2pool = NULL;
    mapping = NULL;
    p_kg = NULL;
}

RuleGenerator::~RuleGenerator()
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
        rel2rules = NULL;
    }
    if (rel2pool != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2pool[r].clear();
        delete [] rel2pool;
        rel2pool = NULL;
    }
    if (mapping != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) mapping[r].clear();
        delete [] mapping;
        mapping = NULL;
    }
    p_kg = NULL;
}

void RuleGenerator::init_knowledge_graph(KnowledgeGraph *_p_kg)
{
    p_kg = _p_kg;
}

void RuleGenerator::set_logic_rules(std::vector<Rule> * _rel2rules)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
    
    rel2rules = new std::vector<Rule> [p_kg->relation_size];
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        rel2rules[r] = _rel2rules[r];
        for (int k = 0; k != int(rel2rules[r].size()); k++)
        {
            rel2rules[r][k].wt.clear();
            rel2rules[r][k].H = 0;
        }
    }
}

std::vector<Rule> *RuleGenerator::get_logic_rules()
{
    return rel2rules;
}

void RuleGenerator::set_pool(std::vector<Rule> * _rel2rules)
{
    if (rel2pool != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2pool[r].clear();
        delete [] rel2pool;
    }
    
    rel2pool = new std::vector<Rule> [p_kg->relation_size];
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        rel2pool[r] = _rel2rules[r];
        for (int k = 0; k != int(rel2pool[r].size()); k++)
        {
            rel2pool[r][k].wt.clear();
            rel2pool[r][k].H = 0;
            rel2pool[r][k].cn = 0;
        }
    }
}

void RuleGenerator::sample_from_pool(int _number, double _temperature)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
    if (mapping != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) mapping[r].clear();
        delete [] mapping;
    }
    
    rel2rules = new std::vector<Rule> [p_kg->relation_size];
    mapping = new std::vector<int> [p_kg->relation_size];
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        std::vector<double> probability;
        double max_val = -1000000, sum_val = 0;
        for (int k = 0; k != int(rel2pool[r].size()); k++)
            max_val = std::max(max_val, rel2pool[r][k].H);
        for (int k = 0; k != int(rel2pool[r].size()); k++)
            sum_val += exp((rel2pool[r][k].H - max_val) / _temperature);
        for (int k = 0; k != int(rel2pool[r].size()); k++)
            probability.push_back(exp((rel2pool[r][k].H - max_val) / _temperature) / sum_val);
        
        for (int k = 0; k != _number; k++)
        {
            double sum_prob = 0, rand_val = double(rand()) / double(RAND_MAX);
            for (int index = 0; index != int(rel2pool[r].size()); index++)
            {
                sum_prob += probability[index];
                if (sum_prob > rand_val)
                {
                    rel2rules[r].push_back(rel2pool[r][index]);
                    mapping[r].push_back(index);
                    break;
                }
            }
        }
    }
}

void RuleGenerator::random_from_pool(int _number)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
    if (mapping != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) mapping[r].clear();
        delete [] mapping;
    }
    
    rel2rules = new std::vector<Rule> [p_kg->relation_size];
    mapping = new std::vector<int> [p_kg->relation_size];
    std::vector<int> rand_index;
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        rand_index.clear();
        for (int k = 0; k != int(rel2pool[r].size()); k++) rand_index.push_back(k);
        std::random_shuffle(rand_index.begin(), rand_index.end());
        for (int k = 0; k != int(rel2pool[r].size()); k++)
        {
            if (k >= _number) break;
            int index = rand_index[k];
            rel2rules[r].push_back(rel2pool[r][index]);
            mapping[r].push_back(index);
        }
    }
}

void RuleGenerator::best_from_pool(int _number)
{
    if (rel2rules != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) rel2rules[r].clear();
        delete [] rel2rules;
    }
    if (mapping != NULL)
    {
        for (int r = 0; r != p_kg->relation_size; r++) mapping[r].clear();
        delete [] mapping;
    }
    
    rel2rules = new std::vector<Rule> [p_kg->relation_size];
    mapping = new std::vector<int> [p_kg->relation_size];
    std::vector<RankListEntry> rank_list;
    RankListEntry entry;
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        rank_list.clear();
        for (int k = 0; k != int(rel2pool[r].size()); k++)
        {
            entry.id = k;
            entry.val = rel2pool[r][k].H;
            rank_list.push_back(entry);
        }
        std::sort(rank_list.begin(), rank_list.end());
        for (int k = 0; k != int(rel2pool[r].size()); k++)
        {
            if (k >= _number) break;
            int index = rank_list[k].id;
            rel2rules[r].push_back(rel2pool[r][index]);
            mapping[r].push_back(index);
        }
    }
}

void RuleGenerator::update(std::vector<Rule> * _rel2rules)
{
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        for (int k = 0; k != int(rel2rules[r].size()); k++) rel2rules[r][k].H = _rel2rules[r][k].H;
        for (int k = 0; k != int(rel2rules[r].size()); k++)
        {
            int index = mapping[r][k];
            rel2pool[r][index].H = (rel2pool[r][index].H * rel2pool[r][index].cn + rel2rules[r][k].H) / (rel2pool[r][index].cn + 1);
            rel2pool[r][index].cn += 1;
        }
    }
}

void RuleGenerator::out_rules(char *file_name, int num_rules)
{
    FILE *fo = fopen(file_name, "wb");
    std::vector<RankListEntry> rank_list;
    RankListEntry entry;
    for (int r = 0; r != p_kg->relation_size; r++)
    {
        rank_list.clear();
        for (int k = 0; k != int(rel2pool[r].size()); k++)
        {
            entry.id = k;
            entry.val = rel2pool[r][k].H;
            rank_list.push_back(entry);
        }
        std::sort(rank_list.begin(), rank_list.end());

        int actual_rules = int(rel2pool[r].size());
        if (num_rules < actual_rules) actual_rules = num_rules;

        //fprintf(fo, "%d %d\n", r, actual_rules);
        for (int k = 0; k != int(rel2pool[r].size()); k++)
        {
            if (k == num_rules) break;
            int index = rank_list[k].id;
            //fprintf(fo, "%d %d %d", k, rel2pool[r][index].r_head, rel2pool[r][index].type);
            fprintf(fo, "%d", rel2pool[r][index].r_head);
            for (int i = 0; i != int(rel2pool[r][index].r_body.size()); i++) fprintf(fo, " %d", rel2pool[r][index].r_body[i]);
            fprintf(fo, " %.16lf", rel2pool[r][index].H);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
}
