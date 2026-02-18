// BGL Louvain benchmark: adjacency_list<setS, vecS>
// Edge container: ordered set. No parallel edges, O(log(degree)) lookup.
#include "bgl_louvain_benchmark.hpp"
int main(int argc, char* argv[]) {
    return bgl_benchmark::run_adjacency_list<boost::setS, boost::vecS>(argc, argv);
}
