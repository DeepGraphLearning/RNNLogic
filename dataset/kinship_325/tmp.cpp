#include <bits/stdc++.h>
using namespace std;
#define IL __inline__ __attribute__((always_inline))

// Change these for every dataset
const int MAX_E = 1000;
const int MAX_R = 200;
const int MAX_RULE_LEN = 3;


map<string, int> entity_id, relation_id;

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

int E, R, mov;

map<pair<int, int>, set<int>> train;
set<int> train_h[MAX_R];
vector<pair<int, int>> train_hr;
FILE* files[MAX_R];

struct Graph {
	vector<int> e[MAX_E][MAX_R];
	vector<pair<int, int>> a[MAX_E];
	IL void add(int h, int r, int t) {
		e[h][r].push_back(t);
		a[h].push_back({r, t});
	}
}	G;

IL void add_data(int h, int r, int t) {
	if(r != R - 1) G.add(h, r, t);
	train[{h, r}].insert(t);
	train_h[r].insert(h);
	train_hr.push_back({h, r});
}

int rcount[MAX_R];
set<vector<int>> sampled[MAX_R];

IL int rinv(int r) {
	if(r == R - 1) return r;
	return r < mov ? r + mov : r - mov;
}

struct Choice {
	int r, q;
	long long c;
};

IL long long randll() {
	return (long long) rand() * RAND_MAX + rand();
}

void verify(int r, vector<int> path, bool force = 0) {
	if(!sampled[r].insert(path).second) return;
	double recall_u = 0, recall_d = 0, prec_u = 0, prec_d = 0;
	for(auto X : train_h[r]) {
		unordered_map<int, long long> pos;
		pos[X] = 1;
		for(auto r : path) {
			unordered_map<int, long long> newpos;
			for(auto pc : pos) {
				int p = pc.first;
				long long c = pc.second;
				for(auto q : G.e[p][r]) newpos[q] += c;
			}
			newpos.swap(pos);
		}
		auto &correct_Y = train[{X, r}];
		for(auto pc : pos) {
			auto p = pc.first;
			auto c = pc.second;
			if(correct_Y.count(p)) {
				prec_u += c;
				recall_u += 1;
			}
			prec_d += c;
		}
		recall_d += correct_Y.size();
	}

	double prec = prec_u / (prec_d + 5), recall = recall_u / recall_d;



	if(force || (prec > 1e-4 && recall > 1e-5)) {
		auto file = files[r];
		for(int i = 0; i < (int) path.size(); ++i)
			fprintf(file, "%d%c", path[i], " \t"[i == (int) path.size() - 1]);	
		fprintf(file, "%.10lf\n", prec);
		rcount[r] += 1;
		if(rcount[r] % 100 == 0) {
			printf("Found r = %d path = ", r); for(auto r : path) printf("%d ", r); printf("%.4lf %.4lf ", prec, recall);
			printf("Written count = %d\n", rcount[r]);
			fflush(file);
		}

	}
}

int main() {
	srand(233666);
	ifstream file;

	file.open("entities.dict");
	for(string id, e; file >> id >> e; ) 
		entity_id[e] = convert<int>(id);
	file.close();

	file.open("relations.dict");
	for(string id, r; file >> id >> r; )
		relation_id[r] = convert<int>(id);
	file.close();

	E = entity_id.size();
	R = relation_id.size();
	mov = R;
	R += mov;
	R += 1;
	printf("E, R = %d, %d\n", E, R);

	// for(int i = 0; i < E; ++i) add_data(i, R - 1, i);

	file.open("train.txt");
	for(string _h, _r, _t; file >> _h >> _r >> _t; ) {
		int h = entity_id[_h];
		int r = relation_id[_r];
		int t = entity_id[_t];
		add_data(h, r, t);
		add_data(t, r + mov, h);
	}
	file.close();

	int MAX_LEN = 3;
	int MAX_LEN_HARD = MAX_RULE_LEN;
	--MAX_LEN;
	printf("total hr = %d\n", int(train_hr.size()));

	for(int r = 0; r < R; ++r) {
		if(r == R - 1) continue;
		files[r] = fopen(("Rules/rules_" + str(r) + ".txt").c_str(), "w");
		sampled[r].insert(vector<int>{r});
	}

	for(int num_rounds = 0; ; ++num_rounds) {
		if(num_rounds % train_hr.size() == 0) {
			random_shuffle(train_hr.begin(), train_hr.end());
			if(MAX_LEN < MAX_LEN_HARD) MAX_LEN += 1;
		}
		const auto cur = train_hr[num_rounds % train_hr.size()];
		const int h = cur.first, r = cur.second;
		const auto& t = train[{h, r}];

		unordered_map<int, long long> path_cnt[MAX_LEN + 1];
		unordered_map<int, vector<Choice>> choices[MAX_LEN + 1];

		for(auto i : t) {
			path_cnt[0][i] = 1;
			choices[0][i].push_back({-1, -1, 1});
		}

		for(int _ = 0; _ < MAX_LEN; ++_) {
			auto &next_cnt = path_cnt[_ + 1];
			auto &next_cho = choices[_ + 1];
			for(auto path : path_cnt[_]) {
				auto i = path.first;
				auto cnt = path.second;
				for(auto edge : G.a[i]) {
					next_cnt[edge.second] += cnt;
					auto &cho = next_cho[edge.second];
					long long last_c = (cho.empty() ? 0 : cho.back().c);
					cho.push_back({rinv(edge.first), i, cnt + last_c});
				}
			}
		}

		vector<int> possible_len;
		for(int i = 1; i <= MAX_LEN; ++i)
			if(path_cnt[i][h]) {
				possible_len.push_back(i);
				if(i <= 3) possible_len.push_back(i);
			}

		if(num_rounds % 1000 == 0 || num_rounds <= 10)
			printf("Work %d = (%d, %d) count = %d |possible_len| = %d\n",
				num_rounds, h, r, rcount[r], int(possible_len.size()));

		for(int i = 0; i < R; ++i) {
			verify(r, vector<int>{i}, 1);
		}
		
		for(int n_sample = (rcount[r] < 100 ? 100 : 25) * possible_len.size(); n_sample--; ) {
			vector<int> path;
			for(int p = h, l = possible_len[rand() % possible_len.size()]; l > 0; --l) {
				auto &cho = choices[l][p];
				long long k = randll() % path_cnt[l][p] + 1;
				int L = 0, R = cho.size() - 1, M;
				while(L < R) {
					M = (L + R) >> 1;
					if(k <= cho[M].c) R = M;
					else L = M + 1;
				}
				if(path.size() > 0u and path.back() == rinv(cho[L].r)) goto skipped;
				path.push_back(cho[L].r);
				p = cho[L].q;
			}
			verify(r, path);
			
			
			skipped:
			continue;
			// return 0;



		}



	}

}