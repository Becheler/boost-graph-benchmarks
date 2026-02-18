// BGL Louvain benchmark: adjacency_list<vecS, vecS>
// Fastest random access, implicit vertex_index.
#include "bgl_louvain_benchmark.hpp"
int main(int argc, char* argv[]) {
    return bgl_benchmark::run_adjacency_list<boost::vecS, boost::vecS>(argc, argv);
}
