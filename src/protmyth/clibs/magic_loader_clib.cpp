// Copyright (c) 2024 Helixon Limited.
//
// This file is a part of ProtMyth and is released under the MIT License.
// Thanks for using ProtMyth!

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;


namespace minimum_cost_flow {
  const int DIST_INF = 1e9, FLOW_INF = 1e9;

  int n_ver;

  class edge_type {
  public:
    int v, flow, cost, inverse_index;

    edge_type(): v(0), flow(0), cost(0), inverse_index(0) {}

    edge_type(int v, int flow, int cost, int inverse_index): v(v), flow(flow), cost(cost), inverse_index(inverse_index) {}
  };
  vector<vector<edge_type> > graph;

  void init(int n_ver) {
    minimum_cost_flow::n_ver = n_ver;
    graph = vector<vector<edge_type> >(n_ver, vector<edge_type>(0));
  }

  void add_edge(int u, int v, int flow, int cost) {
    graph[u].push_back(edge_type(v, flow, cost, graph[v].size()));
    graph[v].push_back(edge_type(u, 0, -cost, graph[u].size() - 1));
  }

  pair<vector<int>, vector<pair<int, int> > > shortest_path(int s_ver) {
    vector<int> dist(n_ver, DIST_INF);
    vector<pair<int, int> > last_edge(n_ver);
    vector<bool> queue_flag(n_ver, false);
    queue<int> queue_ver;

    queue_ver.push(s_ver);
    queue_flag[s_ver] = true;
    dist[s_ver] = 0;
    while (!queue_ver.empty()) {
      int cur_ver = queue_ver.front();
      queue_ver.pop();
      queue_flag[cur_ver] = false;
      for (int i = 0; i < graph[cur_ver].size(); i ++) {
        int next_ver = graph[cur_ver][i].v;
        if (graph[cur_ver][i].flow && dist[cur_ver] + graph[cur_ver][i].cost < dist[next_ver]) {
          dist[next_ver] = dist[cur_ver] + graph[cur_ver][i].cost;
          last_edge[next_ver] = make_pair(cur_ver, i);
          if (!queue_flag[next_ver]) {
            queue_ver.push(next_ver);
            queue_flag[next_ver] = true;
          }
        }
      }
    }
    return make_pair(dist, last_edge);
  }

  pair<int, vector<pair<int, int> > > min_cost_flow(int s_ver, int t_ver) {
    int min_cost = 0;
    while (true) {
      auto sp_ret = shortest_path(s_ver);
      auto dist = sp_ret.first;
      auto last_edge = sp_ret.second;
      if (dist[t_ver] == DIST_INF)
        break;

      int flow = FLOW_INF;
      for (int cur_ver = t_ver; cur_ver != s_ver; cur_ver = last_edge[cur_ver].first)
        flow = min(flow, graph[last_edge[cur_ver].first][last_edge[cur_ver].second].flow);

      min_cost += dist[t_ver] * flow;

      for (int cur_ver = t_ver; cur_ver != s_ver; cur_ver = last_edge[cur_ver].first) {
        graph[last_edge[cur_ver].first][last_edge[cur_ver].second].flow -= flow;
        graph[cur_ver][graph[last_edge[cur_ver].first][last_edge[cur_ver].second].inverse_index].flow += flow;
      }
    }

    vector<pair<int, int> > matching;
    for (int ver = 0; ver < n_ver; ver ++)
      for (auto edge : graph[ver])
        if (edge.cost < 0 && edge.flow == 0)
          matching.push_back(make_pair(ver, edge.v));

    return make_pair(min_cost, matching);
  }
}


const int PARA_LENGTH_LIMIT = 1e4;
int longest_common_sequence(const char key1[PARA_LENGTH_LIMIT], const char key2[PARA_LENGTH_LIMIT]) {
  int len1 = strlen(key1);
  int len2 = strlen(key2);
  vector<vector<int> > f(len1, vector<int>(len2));
  for(int i = 0; i < len1; i ++)
    for(int j = 0; j < len2; j ++) {
      if (key1[i] == key2[j]) {
        f[i][j] = max(f[i][j], 1);
        if (i > 0 && j > 0)
          f[i][j] = max(f[i][j], 1 + f[i - 1][j - 1]);
      }
      if (i > 0)
        f[i][j] = max(f[i][j], f[i - 1][j]);
      if (j > 0)
        f[i][j] = max(f[i][j], f[i][j - 1]);
    }
  return f[len1 - 1][len2 - 1];
}


extern "C" {
  int magic_loader(const int n_keys1, const char keys1[][PARA_LENGTH_LIMIT],
                   const int n_keys2, const char keys2[][PARA_LENGTH_LIMIT],
                   int match1[], int match2[]) {
    minimum_cost_flow::init(n_keys1 + n_keys2 + 2);
    for (int i = 0; i < n_keys1; i ++)
      minimum_cost_flow::add_edge(0, i + 1, 1, 0);
    for (int i = 0; i < n_keys2; i ++)
      minimum_cost_flow::add_edge(n_keys1 + i + 1, n_keys1 + n_keys2 + 1, 1, 0);
    vector<vector<int> > lcs(n_keys1, vector<int>(n_keys2, 0));
    #pragma omp parallel for num_threads(8) schedule(guided)
    for (int i = 0; i < n_keys1; i ++)
      for (int j = 0; j < n_keys2; j ++)
        lcs[i][j] = longest_common_sequence(keys1[i], keys2[j]);
    for (int i = 0; i < n_keys1; i ++)
      for (int j = 0; j < n_keys2; j ++)
        minimum_cost_flow::add_edge(i + 1, n_keys1 + j + 1, 1, -lcs[i][j]);
    auto match = minimum_cost_flow::min_cost_flow(0, n_keys1 + n_keys2 + 1).second;

    int n_match = 0;
    for (auto m : match) {
      match1[n_match] = m.first - 1;
      match2[n_match] = m.second - n_keys1 - 1;
      n_match ++;
    }

    return n_match;
  }
}
