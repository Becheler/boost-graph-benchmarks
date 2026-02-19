// BGL Louvain benchmark: adjacency_matrix, NON-INCREMENTAL quality function
#include "bgl_louvain_benchmark.hpp"
int main(int argc, char* argv[]) {
    return bgl_benchmark::run_adjacency_matrix<
        bgl_benchmark::newman_and_girvan_non_incremental>(argc, argv);
}
