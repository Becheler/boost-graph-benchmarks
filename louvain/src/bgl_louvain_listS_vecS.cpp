// BGL Louvain benchmark: adjacency_list<listS, vecS>
// Edge container: linked list. O(1) edge insert/remove, worse cache locality.
#include "bgl_louvain_benchmark.hpp"
int main(int argc, char* argv[]) {
    return bgl_benchmark::run_adjacency_list<boost::listS, boost::vecS>(argc, argv);
}
