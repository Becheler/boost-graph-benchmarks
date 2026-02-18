// BGL Louvain benchmark: adjacency_matrix
// O(1) edge lookup, O(V^2) space. Best for dense graphs, small N only.
#include "bgl_louvain_benchmark.hpp"
int main(int argc, char* argv[]) {
    return bgl_benchmark::run_adjacency_matrix(argc, argv);
}
