#include <Python.h>
#include "rnnlogic.h"
#include <vector>
#include <string.h>

static PyObject *new_knowledge_graph(PyObject *self, PyObject *args)
{
    char *data_path;
    
    if (!PyArg_ParseTuple(args, "s", &data_path))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    KnowledgeGraph *p_kg = new KnowledgeGraph;
    p_kg->read_data(data_path);
    return PyLong_FromLong(long(p_kg));
}

static PyObject *num_entities(PyObject *self, PyObject *args)
{
    long long_val;
    
    if (!PyArg_ParseTuple(args, "l", &long_val))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(long_val);
    return PyLong_FromLong(long(p_kg->get_entity_size()));
}

static PyObject *num_relations(PyObject *self, PyObject *args)
{
    long long_val;
    
    if (!PyArg_ParseTuple(args, "l", &long_val))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(long_val);
    return PyLong_FromLong(long(p_kg->get_relation_size()));
}

static PyObject *new_rule_miner(PyObject *self, PyObject *args)
{
    long long_val;

    if (!PyArg_ParseTuple(args, "l", &long_val))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    KnowledgeGraph *p_kg = (KnowledgeGraph *)(long_val);
    RuleMiner *p_rm = new RuleMiner;
    p_rm->init_knowledge_graph(p_kg);
    return PyLong_FromLong(long(p_rm));
}

static PyObject *mine_logic_rules(PyObject *self, PyObject *args)
{
    long long_val;
    int max_length, num_threads;
    double portion;

    if (!PyArg_ParseTuple(args, "l|i|d|i", &long_val, &max_length, &portion, &num_threads))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    RuleMiner *p_rm = (RuleMiner *)(long_val);
    p_rm->search(max_length, portion, num_threads);
    return Py_None;
}

static PyObject *get_mined_logic_rules(PyObject *self, PyObject *args)
{
    long long_val;
    double double_val;

    if (!PyArg_ParseTuple(args, "l", &long_val))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    RuleMiner *p_rm = (RuleMiner *)(long_val);
    int num_relations = p_rm->get_relation_size();

    PyObject* all_rules = PyList_New(0);
    for (int r = 0; r != num_relations; r++)
    {
        PyObject* rel_rules = PyList_New(0);
        for (int k = 0; k != int((p_rm->get_logic_rules())[r].size()); k++)
        {
            PyObject* rule = PyList_New(0);

            long_val = (p_rm->get_logic_rules())[r][k].type;
            PyList_Append(rule, PyLong_FromLong(long_val));

            long_val = (p_rm->get_logic_rules())[r][k].r_head;
            PyList_Append(rule, PyLong_FromLong(long_val));

            for (int i = 0; i != int((p_rm->get_logic_rules())[r][k].r_body.size()); i++)
            {
                long_val = (p_rm->get_logic_rules())[r][k].r_body[i];
                PyList_Append(rule, PyLong_FromLong(long_val));
            }

            double_val = (p_rm->get_logic_rules())[r][k].H;
            PyList_Append(rule, PyFloat_FromDouble(double_val));

            double_val = (p_rm->get_logic_rules())[r][k].wt.data;
            PyList_Append(rule, PyFloat_FromDouble(double_val));

            double_val = (p_rm->get_logic_rules())[r][k].prior;
            PyList_Append(rule, PyFloat_FromDouble(double_val));

            PyList_Append(rel_rules, rule);
        }
        PyList_Append(all_rules, rel_rules);
    }
    return all_rules;
}

static PyObject *new_reasoning_predictor(PyObject *self, PyObject *args)
{
    long long_val;

    if (!PyArg_ParseTuple(args, "l", &long_val))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    KnowledgeGraph *p_kg = (KnowledgeGraph *)(long_val);
    ReasoningPredictor *p_rp = new ReasoningPredictor;
    p_rp->init_knowledge_graph(p_kg);
    return PyLong_FromLong(long(p_rp));
}

static PyObject *set_reasoning_predictor(PyObject *self, PyObject *args)
{
    long long_val;
    double double_val;
    PyObject* all_rules;
    PyObject* rel_rules;
    PyObject* single_rule;
    PyObject* item;

    if (!PyArg_ParseTuple(args, "l|O", &long_val, &all_rules))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!PyList_Check(all_rules))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ReasoningPredictor *p_rp = (ReasoningPredictor *)(long_val);
    int num_relations = p_rp->get_relation_size();

    std::vector<Rule> *rel2rules = new std::vector<Rule> [num_relations];
    Rule rule;
    for (int r = 0; r != num_relations; r++)
    {
        rel_rules = PyList_GetItem(all_rules, r);
        if (!PyList_Check(rel_rules))
        {
            printf("Input error!\n");
            Py_INCREF(Py_None);
            return Py_None;
        }

        int num_rules = PyList_Size(rel_rules);
        for (int k = 0; k != num_rules; k++)
        {
            single_rule = PyList_GetItem(rel_rules, k);
            int length = int(PyList_Size(single_rule));

            rule.clear();

            item = PyList_GetItem(single_rule, 0);
            long_val = PyLong_AsLong(item);
            rule.type = int(long_val);

            item = PyList_GetItem(single_rule, 1);
            long_val = PyLong_AsLong(item);
            rule.r_head = int(long_val);

            for (int i = 0; i != int(rule.type); i++)
            {
                item = PyList_GetItem(single_rule, 2 + i);
                long_val = PyLong_AsLong(item);
                rule.r_body.push_back(int(long_val));
            }

            item = PyList_GetItem(single_rule, length - 1);
            double_val = PyFloat_AsDouble(item);
            rule.prior = double_val;

            rel2rules[r].push_back(rule);
        }
    }

    p_rp->set_logic_rules(rel2rules);

    for (int r = 0; r != num_relations; r++) rel2rules[r].clear();
    delete [] rel2rules;
    
    return Py_None;
}

static PyObject *get_assessed_logic_rules(PyObject *self, PyObject *args)
{
    long long_val;
    double double_val;

    if (!PyArg_ParseTuple(args, "l", &long_val))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ReasoningPredictor *p_rp = (ReasoningPredictor *)(long_val);
    int num_relations = p_rp->get_relation_size();

    PyObject* all_rules = PyList_New(0);
    for (int r = 0; r != num_relations; r++)
    {
        PyObject* rel_rules = PyList_New(0);
        for (int k = 0; k != int((p_rp->get_logic_rules())[r].size()); k++)
        {
            PyObject* rule = PyList_New(0);

            long_val = (p_rp->get_logic_rules())[r][k].type;
            PyList_Append(rule, PyLong_FromLong(long_val));

            long_val = (p_rp->get_logic_rules())[r][k].r_head;
            PyList_Append(rule, PyLong_FromLong(long_val));

            for (int i = 0; i != int((p_rp->get_logic_rules())[r][k].r_body.size()); i++)
            {
                long_val = (p_rp->get_logic_rules())[r][k].r_body[i];
                PyList_Append(rule, PyLong_FromLong(long_val));
            }

            double_val = (p_rp->get_logic_rules())[r][k].H;
            PyList_Append(rule, PyFloat_FromDouble(double_val));

            double_val = (p_rp->get_logic_rules())[r][k].wt.data;
            PyList_Append(rule, PyFloat_FromDouble(double_val));

            double_val = (p_rp->get_logic_rules())[r][k].prior;
            PyList_Append(rule, PyFloat_FromDouble(double_val));

            PyList_Append(rel_rules, rule);
        }
        PyList_Append(all_rules, rel_rules);
    }
    return all_rules;
}

static PyObject *train_reasoning_predictor(PyObject *self, PyObject *args)
{
    long long_val;
    int num_threads;
    double learning_rate, weight_decay, temperature, portion;

    if (!PyArg_ParseTuple(args, "l|d|d|d|d|i", &long_val, &learning_rate, &weight_decay, &temperature, &portion, &num_threads))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ReasoningPredictor *p_rp = (ReasoningPredictor *)(long_val);
    p_rp->learn(learning_rate, weight_decay, temperature, portion, num_threads);

    return Py_None;
}

static PyObject *compute_H_score(PyObject *self, PyObject *args)
{
    long long_val;
    int top_k, num_threads;
    double H_temperature, prior_weight, portion;

    if (!PyArg_ParseTuple(args, "l|i|d|d|d|i", &long_val, &top_k, &H_temperature, &prior_weight, &portion, &num_threads))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ReasoningPredictor *p_rp = (ReasoningPredictor *)(long_val);
    p_rp->H_score(top_k, H_temperature, prior_weight, portion, num_threads);

    return Py_None;
}

static PyObject *test_reasoning_predictor(PyObject *self, PyObject *args)
{
    long long_val;
    char *mode;
    int num_threads;

    if (!PyArg_ParseTuple(args, "l|s|i", &long_val, &mode, &num_threads))
    {
        printf("Input error!\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ReasoningPredictor *p_rp = (ReasoningPredictor *)(long_val);

    Result result;
    if (strcmp(mode, "valid") == 0)
    {
        result = p_rp->evaluate(false, num_threads);
    }
    if (strcmp(mode, "test") == 0)
    {
        result = p_rp->evaluate(true, num_threads);
    }

    PyObject* list = PyList_New(0);
    double double_val;

    double_val = result.mr;
    PyList_Append(list, PyFloat_FromDouble(double_val));

    double_val = result.mrr;
    PyList_Append(list, PyFloat_FromDouble(double_val));

    double_val = result.h1;
    PyList_Append(list, PyFloat_FromDouble(double_val));

    double_val = result.h3;
    PyList_Append(list, PyFloat_FromDouble(double_val));

    double_val = result.h10;
    PyList_Append(list, PyFloat_FromDouble(double_val));

    return list;
}

static PyMethodDef PyExtMethods[] =
{
    { "new_knowledge_graph", new_knowledge_graph, METH_VARARGS, "new_knowledge_graph" },
    { "num_entities", num_entities, METH_VARARGS, "num_entities" },
    { "num_relations", num_relations, METH_VARARGS, "num_relations" },
    { "new_rule_miner", new_rule_miner, METH_VARARGS, "new_rule_miner" },
    { "mine_logic_rules", mine_logic_rules, METH_VARARGS, "mine_logic_rules" },
    { "get_mined_logic_rules", get_mined_logic_rules, METH_VARARGS, "get_mined_logic_rules" },
    { "new_reasoning_predictor", new_reasoning_predictor, METH_VARARGS, "new_reasoning_predictor" },
    { "set_reasoning_predictor", set_reasoning_predictor, METH_VARARGS, "set_reasoning_predictor" },
    { "get_assessed_logic_rules", get_assessed_logic_rules, METH_VARARGS, "get_assessed_logic_rules" },
    { "train_reasoning_predictor", train_reasoning_predictor, METH_VARARGS, "train_reasoning_predictor" },
    { "compute_H_score", compute_H_score, METH_VARARGS, "compute_H_score" },
    { "test_reasoning_predictor", test_reasoning_predictor, METH_VARARGS, "test_reasoning_predictor" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef pyrnnlogic =
{
    PyModuleDef_HEAD_INIT,
    "pyrnnlogic",
    "",
    -1,
    PyExtMethods
};

PyMODINIT_FUNC PyInit_pyrnnlogic(void)
{
    return PyModule_Create(&pyrnnlogic);
}