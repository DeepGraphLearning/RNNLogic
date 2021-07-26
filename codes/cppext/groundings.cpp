#ifndef NOBIND
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
namespace py = pybind11;
#endif

#include <cstdio>
#include <map>
#include <vector>
#include <set>
using namespace std;
#define IL inline
#define debug // printf

struct Graph {
	int E, R;
	vector<int> **e;
	vector<pair<int, int>> *a;
	IL void add(int h, int r, int t) {
		e[h][r].push_back(t);
		a[h].push_back({r, t});
	}

	void init(int _E, int _R) {
		clear();
		E = _E;
		R = _R;
		debug("Init: E = %d R = %d\n", E, R);
		e = new vector<int>*[E + 5];
		a = new vector<pair<int, int>>[E + 5];
		for(int i = 0; i < E + 5; ++i)
			e[i] = new vector<int>[R + 5];
	}
	void clear() {
		if(!E || !R) return;
		for(int i = 0; i < E + 5; ++i) 
			delete[] e[i];
		delete[] e;
		delete[] a;
		E = R = 0;
	}

	~Graph() {
		clear();
	}
};

namespace cppext_groundings {
	using ull = unsigned long long;

	template<class T> IL T convert(string s) {
		T ret;
		istringstream in(s);
		in >> ret;
		return ret;
	}
	template<class T> IL string str(T v) {
		ostringstream out;
		out << v;
		return out.str();
	}

	Graph G;

	vector<int> *result_pts;
	vector<long long> *result_cnt;

	IL void add_data(int h, int r, int t) {
		G.add(h, r, t);
	}

	struct GroundingsTask {
		const Graph *G;
		vector<int> q, v;
		vector<int> result_pts;
		int timer;
		GroundingsTask() : timer(0) {}
		GroundingsTask(const Graph& g) : timer(0) {use_graph(g);}
		void use_graph(const Graph& g) {
			G = &g; q.resize(G->E); v.resize(G->E); // c.resize(G->E);
		}
		void run(int h, const vector<int>& path) {
			// int *q = new int[E], *v = new int[E];
			if(q.size() <= path.size() * G->E) 
				q.resize(path.size() * G->E + 1);
			int l = 0, r = 0; v[q[r++] = h] = timer;
			for(auto rel : path) {
				++timer;
				for(int n = r, p; l < n; ++l) {
					p = q[l];
					for(auto k : G->e[p][rel]) 
						if(v[k] != timer) v[q[r++] = k] = timer;

				}
			}
			result_pts = vector<int>(&q[l], &q[r]);
		}
	}	gnd;

	void groundings(int h, const vector<int>& path) {
		gnd.run(h, path);
		result_pts = &(gnd.result_pts);
	}

	struct GroundingsCountTask {
		const Graph *G;
		vector<int> q, v;
		vector<long long> c;
		vector<int> result_pts;
		vector<long long> result_cnt;
		int timer;
		GroundingsCountTask() : timer(0) {}
		GroundingsCountTask(const Graph& g) : timer(0) {use_graph(g);}
		void use_graph(const Graph& g) {
			G = &g; q.resize(G->E); v.resize(G->E); c.resize(G->E);
		}
		void run(int h, const vector<int>& path) {
			if(q.size() <= path.size() * G->E) 
				q.resize(path.size() * G->E + 1);
			int l = 0, r = 0; v[q[r++] = h] = timer; c[h] = 1;
			for(auto rel : path) {
				++timer;
				for(int n = r; l < n; ++l) {
					auto p = q[l]; auto cp = c[p];
					for(auto k : G->e[p][rel]) {
						if(v[k] != timer) { v[q[r++] = k] = timer; c[k] = 0; }
						c[k] += cp;
					}
				}
			}
			result_pts = vector<int>(&q[l], &q[r]);
			vector<long long> cnt;
			for(int i = l; i < r; ++i) cnt.push_back(c[q[i]]);
			result_cnt = cnt;
		}
	}	gndcnt;

	void groundings_count(int h, const vector<int>& path) {
		gndcnt.run(h, path);
		result_pts = &(gndcnt.result_pts);
		result_cnt = &(gndcnt.result_cnt);
	}
}

#ifndef NOBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	using namespace cppext_groundings;
	m.doc() = "Find groundings for KG";
	m.def("init", [&] (int E, int R) {
		// cppext_groundings::E = E;
		// cppext_groundings::R = R;

		G.clear();
		G.init(E, R);
		gnd.use_graph(G);
		gndcnt.use_graph(G);

	}, py::arg("E"), py::arg("R"));

	m.def("add", add_data, py::arg("h"), py::arg("r"), py::arg("t"));

	m.def("calc", groundings, py::arg("h"), py::arg("path"));
	m.def("calc_count", groundings_count, py::arg("h"), py::arg("path"));

	m.def("result_pts", [&] () {return *result_pts;});
	m.def("result_cnt", [&] () {return *result_cnt;});

	// m.def("test", [&] () {map<int,int> a; a[1]=2; return a;});
}

#endif
#undef IL
#undef debug