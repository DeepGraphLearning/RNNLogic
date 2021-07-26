#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include <cmath>

#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

using namespace std;


const int MAX_N = 512 + 1;
const double inf = 1e100;
#define append push_back

namespace MCMF {
	int Q[MAX_N * MAX_N << 3];
	double D[MAX_N];
	bool V[MAX_N];

	bool need_backup;

	void backup(struct Adj* ptr);
	struct Adj{
		int to, flow;
		double cost; 
		Adj *next, *rev;

		void augment(int v) {
			backup(this);
			flow -= v;
			backup(rev);
			rev->flow += v;
		}
	};

	vector<pair<Adj*, Adj>> backup_record;
	void backup(Adj* k) {
		if(need_backup) {
			backup_record.append({k, *k});
		}
	}
	void start_backup() {
		need_backup = 1;
		backup_record.clear();
	}
	void restore() {
		need_backup = 0;
		for(auto x : backup_record) 
			*(x.first) = x.second;
		backup_record.clear();
	}


	struct Graph {
		Adj *fir[MAX_N], *cur[MAX_N], mem[MAX_N * MAX_N * 2 + MAX_N * 4], *tot;

		int n, S, T;

		void init() {
			tot = mem;
			fill(fir, fir + n, (Adj*) NULL);
		}

		void add(int a, int b, int f, double c) {
			*++tot = {b, f, c, fir[a]}, fir[a] = tot;
			*++tot = {a, 0, -c, fir[b]}, fir[b] = tot;
			fir[a]->rev = fir[b];
			fir[b]->rev = fir[a];
		}
		int dfs(int i, int m)
		{
			if(i == T) return m;
			V[i] = 1;
			int u = 0, t;
			double d = D[i];
			for(Adj *&k = cur[i]; k; k = k->next)
				if(k->flow && !V[k->to] && fabs(d + k->cost - D[k->to]) < 1e-6) {
					t = dfs(k->to, min(m - u, k->flow));
					u += t; k->augment(t);
					if(u == m) break;
				}
			V[i] = 0;
			return u;
		}

		double augment(int lim_flow = 1e9, int S1 = -1, int T1 = -1)
		{	
			int S0 = S, T0 = T;
			if(S1 != -1) S = S1;
			if(T1 != -1) T = T1;

			int flow = 0; double cost = 0;
			while(flow < lim_flow)
			{
				int l = MAX_N, r = MAX_N, p;
				double d;
				const Adj *k;
				fill(D, D + n, inf);
				D[Q[r++] = S] = 0;
				while(l != r)
				{
					V[p = Q[l++]] = 0, d = D[p];
					for(k = fir[p]; k; k = k->next)
						if(k->flow && D[k->to] > d + k->cost + 1e-6)
						{
							D[k->to] = d + k->cost;
							if(!V[k->to]) 
								V[Q[(l != r && D[k->to] < D[Q[l]]) ? --l : r++] = k->to] = 1;	
						}
				}
				// exit(0);
				if(D[T] >= 0.5 * inf) break;
				copy(fir, fir + n, cur);
				int new_flow = dfs(S, lim_flow - flow);
				flow += new_flow;
				cost += new_flow * D[T];
				// printf("augment flow = %d cost = %.4lf\n", flow, cost);
			}
			S = S0;
			T = T0;
			return cost;
		}
	}	G;
}

namespace cppext {
	double a[MAX_N][MAX_N];

	int nl, nr, movl, movr;
	void print() {
		using namespace MCMF;
		for(auto kl = G.fir[G.S]; kl; kl = kl->next) {
			const int i = kl->to - movl;
			for(auto km = G.fir[movl + i]; km; km = km->next) if(km->to != G.S) {
				const int j = km->to - movr;
				if(km->rev->flow == 0) continue;
				printf("pair %d %d %.4lf\n", i, j, a[i][j]);
			}
		}
	}


	vector<vector<double>> score(vector<int> cntl, vector<int> cntr, vector<vector<double>> _a) {
		nl = _a.size(), nr = _a[0].size(), movl = 0, movr = nl;
		for(int i = 0; i < nl; ++i) {
			assert(int(_a[i].size()) == nr);
			copy(_a[i].begin(), _a[i].end(), a[i]);
		}

		vector<vector<double>> ans(nl);
		assert((int) cntl.size() == nl);
		assert((int) cntr.size() == nr);

		
		using namespace MCMF;
		G.n = nl + nr;
		G.S = G.n++;
		G.T = G.n++;
		G.init();


		for(int i = 0; i < nl; ++i) G.add(G.S, movl + i, cntl[i], 0.0);
		for(int j = 0; j < nr; ++j) G.add(movr + j, G.T, cntr[j], 0.0);

		for(int i = 0; i < nl; ++i)
			for(int j = 0; j < nr; ++j) 
				G.add(movl + i, movr + j, 1, a[i][j]);

		need_backup = 0;
		double ans_match = G.augment();

		for(auto kl = G.fir[G.S]; kl; kl = kl->next) {
			const int i = kl->to - movl;
			ans[i].resize(nr);

			for(auto km = G.fir[movl + i]; km; km = km->next) if(km->to != G.S) {
				const int j = km->to - movr;
				if(km->rev->flow == 0) continue;
				// matched at least once
				ans[i][j] = ans_match;
			}
			for(auto km = G.fir[movl + i]; km; km = km->next) if(km->to != G.S) {
				const int j = km->to - movr;
				if(km->rev->flow > 0) continue;
				for(auto kr = G.fir[movr + j]; kr; kr = kr->next) if(kr->to == G.T) {
					start_backup();
					backup(km);
					backup(km->rev);
					km->flow = 0;
					km->rev->flow = 0;
					ans[i][j] = ans_match + km->cost + G.augment(1, movr + j, movl + i);
					restore();
					break;
				}
			}
		}
		return ans;
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score", cppext::score, "score");
}

#if 0

int main() {
	int N, M;
	while(cin >> N >> M) {
		// fprintf(stderr, "batch N = %d M = %d\n", N, M);
		vector<vector<double>> a;
		vector<int> cntl, cntr;
		for(int i = 0; i < N; ++i) {int x; cin >> x; cntl.append(x);}
		for(int i = 0; i < M; ++i) {int x; cin >> x; cntr.append(x);}
		for(int i = 0; i < N; ++i) {
			vector<double> b;
			for(int j = 0; j < M; ++j) {double x; cin >> x; b.append(x);}
			a.append(b);
		}
		auto b = cppext::calc(cntl, cntr, a);
		for(int i = 0; i < N; ++i) {
			for(int j = 0; j < M; ++j) printf("%.4lf ", b[i][j]);
			puts("");
		}
	}

}


#endif
