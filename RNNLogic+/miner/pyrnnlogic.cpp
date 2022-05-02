#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
namespace py = pybind11;

#include "rnnlogic.h"
#include <vector>

void *new_knowledge_graph(char *data_path)
{
	KnowledgeGraph *p_kg = new KnowledgeGraph;
    p_kg->read_data(data_path);
    return (void *)(p_kg);
}

int num_entities(void *pt)
{   
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    return p_kg->get_entity_size();
}

int num_relations(void *pt)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    return p_kg->get_relation_size();
}

int num_train_triplets(void *pt)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    return p_kg->get_train_size();
}

int num_valid_triplets(void *pt)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    return p_kg->get_valid_size();
}

int num_test_triplets(void *pt)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    return p_kg->get_test_size();
}

void *new_rule_miner(void *pt)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    RuleMiner *p_rm = new RuleMiner;
    p_rm->init_knowledge_graph(p_kg);
    return (void *)(p_rm);
}

void run_rule_miner(void *pt, int max_length, double portion, int num_threads)
{
    RuleMiner *p_rm = (RuleMiner *)(pt);
    p_rm->search(max_length, portion, num_threads);
    return;
}

std::vector< std::vector<int> > get_logic_rules(void *pt)
{
    RuleMiner *p_rm = (RuleMiner *)(pt);
    std::vector< std::vector<int> > rules;
    std::vector<int> rule;

    int num_relations = p_rm->get_relation_size();
    for (int r = 0; r != num_relations; r++)
    {
        for (int k = 0; k != int((p_rm->get_logic_rules())[r].size()); k++)
        {
            rule.clear();
            rule.push_back((p_rm->get_logic_rules())[r][k].r_head);
            for (int i = 0; i != int((p_rm->get_logic_rules())[r][k].r_body.size()); i++)
            {
                rule.push_back((p_rm->get_logic_rules())[r][k].r_body[i]);
            }
            rules.push_back(rule);
        }
    }
    return rules;
}

void *new_reasoning_predictor(void *pt)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    ReasoningPredictor *p_rp = new ReasoningPredictor;
    p_rp->init_knowledge_graph(p_kg);
    return (void *)(p_rp);
}

void load_reasoning_predictor(void *pt, char *file_name)
{
    ReasoningPredictor *p_rp = (ReasoningPredictor *)(pt);
    p_rp->in_rules(file_name);
    return;
}

bool check_valid(void *pt, int h, int r, int t)
{
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(pt);
    Triplet triplet;
    triplet.h = h; triplet.r = r; triplet.t = t;
    return p_kg->check_true(triplet);
}

std::vector<int> get_data(void *pt, char *mode, double portion, int num_threads)
{
    ReasoningPredictor *p_rp = (ReasoningPredictor *)(pt);
    std::vector<int> data;

    if (strcmp(mode, "train") == 0)
    {
        p_rp->out_train(&data, portion, num_threads);
    }
    if (strcmp(mode, "valid") == 0)
    {
        p_rp->out_test(&data, false, num_threads);
    }
    if (strcmp(mode, "test") == 0)
    {
        p_rp->out_test(&data, true, num_threads);
    }

    return data;
}

std::vector<int> get_data_single(void *pt, char *mode, int h, int r, int t)
{
    ReasoningPredictor *p_rp = (ReasoningPredictor *)(pt);
    std::vector<int> data;

    if (strcmp(mode, "train") == 0)
    {
        p_rp->out_train_single(h, r, t, &data);
    }
    else
    {
        p_rp->out_test_single(h, r, t, &data);
    }

    return data;
}

std::vector<int> get_count(void *pt, char *mode, int num_threads)
{
    ReasoningPredictor *p_rp = (ReasoningPredictor *)(pt);
    std::vector<int> data;

    if (strcmp(mode, "valid") == 0)
    {
        p_rp->out_test_count(&data, false, num_threads);
    }
    if (strcmp(mode, "test") == 0)
    {
        p_rp->out_test_count(&data, true, num_threads);
    }

    return data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "This is pyrnnlogic";
    m.def("new_knowledge_graph", new_knowledge_graph, py::arg("data_path"));
    m.def("num_entities", num_entities, py::arg("pt"));
    m.def("num_relations", num_relations, py::arg("pt"));
    m.def("num_train_triplets", num_train_triplets, py::arg("pt"));
    m.def("num_valid_triplets", num_valid_triplets, py::arg("pt"));
    m.def("num_test_triplets", num_test_triplets, py::arg("pt"));
    m.def("new_rule_miner", new_rule_miner, py::arg("pt"));
    m.def("run_rule_miner", run_rule_miner, py::arg("pt"), py::arg("max_length"), py::arg("portion"), py::arg("num_threads"));
    m.def("get_logic_rules", get_logic_rules, py::arg("pt"));
    m.def("new_reasoning_predictor", new_reasoning_predictor, py::arg("pt"));
    m.def("load_reasoning_predictor", load_reasoning_predictor, py::arg("pt"), py::arg("file_name"));
    m.def("check_valid", check_valid, py::arg("pt"), py::arg("h"), py::arg("r"), py::arg("t"));
    m.def("get_data", get_data, py::arg("pt"), py::arg("mode"), py::arg("portion"), py::arg("num_threads"));
    m.def("get_data_single", get_data_single, py::arg("pt"), py::arg("mode"), py::arg("h"), py::arg("r"), py::arg("t"));
    m.def("get_count", get_count, py::arg("pt"), py::arg("mode"), py::arg("num_threads"));
}
