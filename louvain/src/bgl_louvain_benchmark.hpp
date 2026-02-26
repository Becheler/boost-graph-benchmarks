// bgl_louvain_benchmark.hpp
// Factored template for benchmarking BGL Louvain across graph data structures.
//
// Usage (thin .cpp wrappers):
//   #include "bgl_louvain_benchmark.hpp"
//   int main(int argc, char* argv[]) {
//       return bgl_benchmark::run_adjacency_list<boost::vecS, boost::vecS>(argc, argv);
//   }

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/louvain_clustering.hpp>
#include <boost/property_map/property_map.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <type_traits>
#include <vector>

namespace bgl_benchmark {

// ── Non-incremental quality function wrapper ────────────────────────────
// Delegates to newman_and_girvan::quality() but intentionally omits
// gain/remove/insert so the SFINAE trait resolves to std::false_type,
// forcing the non-incremental (full-recomputation) code path.

struct newman_and_girvan_non_incremental {
    template <typename Graph, typename CommunityMap, typename WeightMap>
    static inline typename boost::property_traits<WeightMap>::value_type
    quality(const Graph& g, const CommunityMap& communities, const WeightMap& weights) {
        return boost::newman_and_girvan::quality(g, communities, weights);
    }
};

// Helpers

/// Parse command-line arguments common to all variants.
inline bool parse_args(int argc, char* argv[], std::string& filename, unsigned int& seed, double& epsilon) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> [seed] [epsilon]\n";
        return false;
    }
    filename = argv[1];
    seed = (argc > 2) ? static_cast<unsigned int>(std::atoi(argv[2])) : 42;
    epsilon = (argc > 3) ? std::atof(argv[3]) : 0.0;
    return true;
}

/// Open an edge-list file and read the header (vertex/edge counts).
inline bool open_graph_file(const std::string& filename, std::ifstream& infile, std::size_t& vertices, std::size_t& edges_count) {
    infile.open(filename);
    if (!infile) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return false;
    }
    infile >> vertices >> edges_count;
    return true;
}

/// Read edges from an already-opened stream.
/// `add_edge_fn(u, v, w)` is called for each edge.
template <typename AddEdgeFn>
void read_edges(std::ifstream& infile, std::size_t edges_count, AddEdgeFn&& add_edge_fn) {
    for (std::size_t i = 0; i < edges_count; ++i) {
        std::size_t u, v;
        double w = 1.0;
        infile >> u >> v;
        if (infile.peek() != '\n' && infile.peek() != EOF)
            infile >> w;
        add_edge_fn(u, v, w);
    }
}

/// Emit timing lines on stderr.
inline void emit_timing(double load_time, double louvain_time) {
    std::cerr << "LOAD_TIME: " << load_time << "\n";
    std::cerr << "LOUVAIN_TIME: " << louvain_time << "\n";
}

// adjacency_list benchmark (template on OutEdgeList x VertexList selectors)

template <typename OutEdgeList, typename VertexList,
          typename QualityFunction = boost::newman_and_girvan>
int run_adjacency_list(int argc, char* argv[]) {

    std::string filename;
    unsigned int seed;
    double epsilon;
    if (!parse_args(argc, argv, filename, seed, epsilon))
        return 1;

    // Type machinery
    constexpr bool is_vecS_vertex = std::is_same_v<VertexList, boost::vecS>;

    using EdgeProp = boost::property<boost::edge_weight_t, double>;

    // vecS vertex: implicit vertex_index, no vertex property needed.
    // listS vertex: explicit vertex_index property required.
    using VertexProp = std::conditional_t<
        is_vecS_vertex,
        boost::no_property,
        boost::property<boost::vertex_index_t, std::size_t>>;

    using Graph = boost::adjacency_list<
        OutEdgeList, VertexList, boost::undirectedS, VertexProp, EdgeProp>;
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;

    // Load graph
    auto load_start = std::chrono::high_resolution_clock::now();

    std::ifstream infile;
    std::size_t vertices, edges_count;
    if (!open_graph_file(filename, infile, vertices, edges_count))
        return 1;

    Graph g;
    std::vector<Vertex> verts; // used only when VertexList == listS

    if constexpr (is_vecS_vertex) {
        for (std::size_t i = 0; i < vertices; ++i)
            boost::add_vertex(g);
    } else {
        verts.resize(vertices);
        for (std::size_t i = 0; i < vertices; ++i)
            verts[i] = boost::add_vertex(VertexProp(i), g);
    }

    read_edges(infile, edges_count,
               [&](std::size_t u, std::size_t v, double w) {
                   if constexpr (is_vecS_vertex)
                       boost::add_edge(u, v, w, g);
                   else
                       boost::add_edge(verts[u], verts[v], EdgeProp(w), g);
               });

    auto load_end = std::chrono::high_resolution_clock::now();
    double load_time =
        std::chrono::duration<double>(load_end - load_start).count();

    // Run Louvain
    std::minstd_rand gen(seed);

    if constexpr (is_vecS_vertex) {
        boost::vector_property_map<Vertex> communities;

        auto t0 = std::chrono::high_resolution_clock::now();
        double Q = boost::louvain_clustering<QualityFunction>(g, communities, boost::get(boost::edge_weight, g), gen, epsilon, epsilon);
        auto t1 = std::chrono::high_resolution_clock::now();

        emit_timing(load_time, std::chrono::duration<double>(t1 - t0).count());

        std::cout << Q << "\n";
        for (std::size_t i = 0; i < vertices; ++i) {
            if (i > 0) std::cout << " ";
            std::cout << boost::get(communities, i);
        }
        std::cout << "\n";
    } else {
        std::map<Vertex, Vertex> comm_store;
        boost::associative_property_map<std::map<Vertex, Vertex>> communities(comm_store);

        auto t0 = std::chrono::high_resolution_clock::now();
        double Q = boost::louvain_clustering<QualityFunction>(g, communities, boost::get(boost::edge_weight, g), gen, epsilon, epsilon);
        auto t1 = std::chrono::high_resolution_clock::now();

        emit_timing(load_time, std::chrono::duration<double>(t1 - t0).count());

        std::cout << Q << "\n";

        // Collect community labels in vertex-index order
        std::vector<Vertex> comm_by_idx(vertices);
        for (std::size_t i = 0; i < vertices; ++i)
            comm_by_idx[i] = boost::get(communities, verts[i]);

        // Renumber to 0..k-1 for consistent output
        std::map<Vertex, std::size_t> comm_renumber;
        std::size_t next_id = 0;
        for (std::size_t i = 0; i < vertices; ++i) {
            if (comm_renumber.find(comm_by_idx[i]) == comm_renumber.end())
                comm_renumber[comm_by_idx[i]] = next_id++;
        }
        for (std::size_t i = 0; i < vertices; ++i) {
            if (i > 0) std::cout << " ";
            std::cout << comm_renumber[comm_by_idx[i]];
        }
        std::cout << "\n";
    }

    return 0;
}

// adjacency_matrix benchmark

template <typename QualityFunction = boost::newman_and_girvan>
int run_adjacency_matrix(int argc, char* argv[]) {

    std::string filename;
    unsigned int seed;
    double epsilon;
    if (!parse_args(argc, argv, filename, seed, epsilon))
        return 1;

    using EdgeProp = boost::property<boost::edge_weight_t, double>;
    using Graph = boost::adjacency_matrix<boost::undirectedS, boost::no_property, EdgeProp>;
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;

    // Load graph
    auto load_start = std::chrono::high_resolution_clock::now();

    std::ifstream infile;
    std::size_t vertices, edges_count;
    if (!open_graph_file(filename, infile, vertices, edges_count))
        return 1;

    Graph g(vertices);

    read_edges(infile, edges_count,
               [&](std::size_t u, std::size_t v, double w) {
                   boost::add_edge(u, v, EdgeProp(w), g);
               });

    auto load_end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(load_end - load_start).count();

    // Run Louvain
    boost::vector_property_map<Vertex> communities;
    std::minstd_rand gen(seed);

    auto t0 = std::chrono::high_resolution_clock::now();
    double Q = boost::louvain_clustering<QualityFunction>(g, communities, boost::get(boost::edge_weight, g), gen, epsilon, epsilon);
    auto t1 = std::chrono::high_resolution_clock::now();

    emit_timing(load_time, std::chrono::duration<double>(t1 - t0).count());

    std::cout << Q << "\n";
    for (std::size_t i = 0; i < vertices; ++i) {
        if (i > 0) std::cout << " ";
        std::cout << boost::get(communities, i);
    }
    std::cout << "\n";

    return 0;
}

} // namespace bgl_benchmark
