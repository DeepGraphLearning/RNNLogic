#ifndef NOBIND
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
namespace py = pybind11;
#endif

#include <cstdio>
#include <map>
#include <vector>
#include <set>
#include <thread>
#include <unordered_map>
#include <random>
// #include <mutex>
#include <shared_mutex>
#include <ctime>
#include <cstdlib>
using namespace std;

#ifndef NOBIND
	#define NOBIND
	#include "groundings.cpp"
	#undef NOBIND
#else
	#include "groundings.cpp"
#endif

using cppext_groundings::GroundingsCountTask;

#define IL inline
#define debug // printf
#define ckpt() //fprintf(stderr, "Thread %d Checkpoint: %d\n", id, __LINE__)
#define ckpt_lock() //fprintf(stderr, "Thread %d Lock: %d\n", id, __LINE__)
#define ckpt_unlock() //fprintf(stderr, "Thread %d Unlock: %d\n", id, __LINE__)

namespace cppext_rule_sample {
	using ull = unsigned long long;

	Graph G, Ginv;
	int E, R;
	void add_data(int h, int r, int t) {
		G.add(h, r, t);
		Ginv.add(t, r, h);
	}


	int MAX_LEN, print_round, num_round;
	map<int, int> len;
	vector<pair<int, vector<int>>> tri;


	map<vector<int>, pair<double, double>> rules;
	shared_mutex rules_mutex;

	struct Choice {
		int r, q;
		long long c;
	};

	unsigned long long seed(int id = 1) {
		char *a = new char, *b = new char;
		auto ret = b - a;
		delete a;
		delete b;
		ret = ret + 1004535809ll * time(0) + clock();
		while(id--) ret = ret * (1 << 16 | 3) + 33333331;
		return ret;
	}

	void work(const int id, GroundingsCountTask gct) {
		auto sd = seed(id);
		// printf("Thread %d: Start seed = %llX.\n", id, sd);
		ckpt();
		default_random_engine e(sd);
		uniform_int_distribution<long long> rand(0, 1ll << 62);

		ckpt();
		// printf("num_round = %d MAX_LEN = %d\n", num_round, MAX_LEN);
		for(int round = 1; round <= num_round; ++round) {
			auto &cur = tri[rand(e) % tri.size()];
			auto h = cur.first;
			auto t = cur.second;

			vector<unordered_map<int, long long>> path_cnt(MAX_LEN + 1);
			vector<unordered_map<int, vector<Choice>>> choices(MAX_LEN + 1);

			ckpt();
			for(auto i : t) {
				path_cnt[0][i] = 1;
				choices[0][i].push_back({-1, -1, 1});
			} 

			ckpt();
			for(int _ = 0; _ < MAX_LEN; ++_) {
				auto &next_cnt = path_cnt[_ + 1];
				auto &next_cho = choices[_ + 1];
				for(auto path : path_cnt[_]) {
					auto i = path.first;
					auto cnt = path.second;
					for(auto edge : Ginv.a[i]) {
						next_cnt[edge.second] += cnt;
						auto &cho = next_cho[edge.second];
						long long last_c = (cho.empty() ? 0 : cho.back().c);
						cho.push_back({edge.first, i, cnt + last_c});
					}
				}
			}

			ckpt();
			for(auto lp : len) {
				int len = lp.first, cnt = lp.second;
				if(path_cnt[len][h] == 0)
					continue;
				for(int _c = 0; _c < cnt; ++_c) {
					// printf("iter _c = %d\n", _c);
					ckpt();
					vector<int> path;
					for(int p = h, l = len; l > 0; --l) {
						auto &cho = choices[l][p];
						long long k = rand(e) % path_cnt[l][p] + 1;
						int L = 0, R = cho.size() - 1, M;
						while(L < R) {
							M = (L + R) >> 1;
							if(k <= cho[M].c) R = M;
							else L = M + 1;
						}
						path.push_back(cho[L].r);
						p = cho[L].q;
					}

					ckpt();
					// ckpt_lock();
					{
						shared_lock<shared_mutex> lock(rules_mutex);
						if(rules.count(path)) continue;
					}
					auto iter = rules.end();
					{
						unique_lock<shared_mutex> lock(rules_mutex);
						iter = rules.insert({path, {0.0, 0.0}}).first;
					}

					ckpt();
					double recall_u = 0, recall_d = 0, prec_u = 0, prec_d = 0;
					for(auto& t : tri) {
						int X = t.first;
						set<int> Y(t.second.begin(), t.second.end());
						recall_d += Y.size();

						// ckpt();
						gct.run(X, path);

						// ckpt();
						for(int i = 0; i < (int) gct.result_pts.size(); ++i) {
							auto p = gct.result_pts[i];
							auto c = gct.result_cnt[i];
							prec_d += c;
							if(Y.count(p)) {
								prec_u += c;
								recall_u += 1;
							}
						}
					}
					double prec = prec_u / max(0.001, prec_d);
					double recall = recall_u / max(0.001, recall_d);
					iter->second = {prec, recall};

					// printf("found _c = %d path = ", _c);
					// for(auto x : path) printf("%d ", x);
					// printf("recall = %.4lf prec = %.4lf\n", recall, prec);
				}
			}

			ckpt();
			if(round % print_round == 0) 
				printf("Thread %d: Round %d/%d.\n", id, round, num_round);
		}
		// printf("Thread %d: Done.\n", id);

	}

	vector<pair<vector<int>, pair<double, double>>>
	run(int r, map<int, int> length_time, int num_samples, 
		int num_threads, double samples_per_print) {
		// printf("Run called\n");
		rules.clear();
		tri.clear();
		for(int i = 0; i < E; ++i)
			if(!G.e[i][r].empty())
				tri.push_back({i, G.e[i][r]});

		// printf("tri.size() = %d\n", tri.size());
		if(!tri.empty()) {
			len = length_time;
			MAX_LEN = 0;
			for(auto a : len) MAX_LEN = max(MAX_LEN, a.first);
			num_round = num_samples / num_threads;
			print_round = samples_per_print;

			if(num_threads > 1) {
				vector<thread> th(num_threads);
				for(int i = 0; i < num_threads; ++i)
					th[i] = thread(work, i + 1, GroundingsCountTask(G));
				for(int i = 0; i < num_threads; ++i)
					th[i].join();
			} else {
				work(1, GroundingsCountTask(G));
			}
			// printf("All Done!\n");
		}

		return vector<pair<vector<int>, pair<double, double>>>(rules.begin(), rules.end());
	}

}

#ifndef NOBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	using namespace cppext_rule_sample;
	m.doc() = "Sample Rules for KG";

	m.def("init", [&] (int E, int R) {
		cppext_rule_sample::E = E;
		cppext_rule_sample::R = R;
		G.clear();
		G.init(E, R);
		Ginv.clear();
		Ginv.init(E, R);

	}, py::arg("E"), py::arg("R"));

	m.def("add", add_data, py::arg("h"), py::arg("r"), py::arg("t"));

	m.def("run", run, py::arg("r"), py::arg("length_time"), py::arg("num_samples"),
		py::arg("num_threads"), py::arg("samples_per_print"));
	
}

#endif
#undef IL
#undef debug
#undef ckpt