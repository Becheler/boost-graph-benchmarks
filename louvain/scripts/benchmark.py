#!/usr/bin/env python3
"""
Benchmark suite for Louvain community detection.

Produces two unified CSVs that cover **every** implementation
(NetworkX, igraph, genlouvain, and all four BGL graph-type variants):

  results/correctness.csv   — per-trial modularity & community count
  results/runtime.csv       — mean/std runtime & community count at scale
"""

import argparse
import networkx as nx
import igraph as ig
import subprocess
import tempfile
import os
import time
import pandas as pd
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────

ALL_BGL_VARIANTS = {
    'BGL vecS/vecS':  './build/bgl_louvain_vecS_vecS',
    'BGL listS/vecS': './build/bgl_louvain_listS_vecS',
    'BGL setS/vecS':  './build/bgl_louvain_setS_vecS',
    'BGL adj_matrix': './build/bgl_louvain_matrix',
}

MATRIX_MAX_NODES = 5000  # adjacency_matrix is O(V^2) memory

ALL_BGL_VARIANTS_NOINC = {
    'BGL vecS/vecS (non-inc)':  './build/bgl_louvain_vecS_vecS_noinc',
    'BGL listS/vecS (non-inc)': './build/bgl_louvain_listS_vecS_noinc',
    'BGL setS/vecS (non-inc)':  './build/bgl_louvain_setS_vecS_noinc',
    'BGL adj_matrix (non-inc)': './build/bgl_louvain_matrix_noinc',
}

# Mapping from non-inc label to its incremental counterpart
NOINC_TO_INC = {
    'BGL vecS/vecS (non-inc)':  'BGL vecS/vecS',
    'BGL listS/vecS (non-inc)': 'BGL listS/vecS',
    'BGL setS/vecS (non-inc)':  'BGL setS/vecS',
    'BGL adj_matrix (non-inc)': 'BGL adj_matrix',
}


def write_edgelist(G, filename):
    """Write graph to edge list format."""
    if not all(isinstance(n, int) for n in G.nodes()):
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

    with open(filename, 'w') as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        for u, v in G.edges():
            weight = G[u][v].get('weight', 1.0)
            f.write(f"{u} {v} {weight}\n")


def genlouvain_available():
    """Check if gen-louvain binaries are present in the build directory."""
    return all(
        os.path.exists(f'./build/genlouvain_{b}')
        for b in ['convert', 'louvain', 'hierarchy']
    )


def run_genlouvain(edgelist_file, seed=42):
    """Run genlouvain implementation. Returns (time, Q, n_communities)."""
    build_dir = './build'

    if not genlouvain_available():
        return None, None, None

    temp_txt = edgelist_file.replace('.txt', '_genlouvain.txt')
    temp_bin = edgelist_file.replace('.txt', '.bin')
    temp_tree = temp_bin + '.tree'

    try:
        with open(edgelist_file, 'r') as f_in:
            lines = f_in.readlines()
            with open(temp_txt, 'w') as f_out:
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        f_out.write(f"{parts[0]} {parts[1]}\n")

        subprocess.run(
            [f'{build_dir}/genlouvain_convert', '-i', temp_txt, '-o', temp_bin],
            capture_output=True, text=True, timeout=30, check=True,
        )

        result = subprocess.run(
            [f'{build_dir}/genlouvain_louvain', temp_bin, '-l', '-1', '-q', '0'],
            capture_output=True, text=True, timeout=300, check=True,
        )

        louvain_time = None
        Q = None
        for line in result.stderr.strip().split('\n'):
            if line.startswith('LOUVAIN_TIME:'):
                louvain_time = float(line.split(':')[1].strip())
            else:
                try:
                    Q = float(line.strip())
                except ValueError:
                    pass

        with open(temp_tree, 'w') as f:
            f.write(result.stdout)

        result = subprocess.run(
            [f'{build_dir}/genlouvain_hierarchy', temp_tree, '-n'],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().split('\n')
        num_levels = int(lines[0].split(':')[1].strip())

        result = subprocess.run(
            [f'{build_dir}/genlouvain_hierarchy', temp_tree, '-l', str(num_levels - 1)],
            capture_output=True, text=True, timeout=10,
        )

        partition_dict = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        partition_dict[int(parts[0])] = int(parts[1])
                    except ValueError:
                        continue

        num_communities = len(set(partition_dict.values()))

        for f in [temp_txt, temp_bin, temp_tree]:
            if f and os.path.exists(f):
                os.remove(f)

        return louvain_time, Q, num_communities
    except Exception:
        for f in [temp_txt, temp_bin, temp_tree]:
            if f and os.path.exists(f):
                os.remove(f)
        return None, None, None


def run_bgl_exe(exe, temp_file, seed='42', timeout=900):
    """Run a BGL executable and parse LOUVAIN_TIME, modularity, and partition."""
    try:
        result = subprocess.run(
            [exe, temp_file, str(seed)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None, None, None

        louvain_time = None
        for line in result.stderr.strip().split('\n'):
            if line.startswith('LOUVAIN_TIME:'):
                louvain_time = float(line.split(':')[1].strip())

        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            Q = float(lines[0])
            partition = [int(x) for x in lines[1].split()]
            return louvain_time, Q, partition
    except Exception:
        pass
    return None, None, None


def _remap_graph(G):
    """Ensure all node labels are contiguous ints starting at 0."""
    G = nx.Graph(G)  # strip multi-edge / directed
    if not all(isinstance(n, int) for n in G.nodes()):
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    return G


# ── correctness ─────────────────────────────────────────────────────────

def run_benchmark_correctness(n_trials=100):
    """Run correctness benchmark on standard graphs.

    Every implementation (NetworkX, igraph, genlouvain, and all four BGL
    graph variants) is tested on every graph.

    Output: results/correctness.csv
        Columns: Graph, Implementation, Modularity, Communities
    """
    print("Correctness Benchmark")
    print("=" * 60)

    test_graphs = {
        'Karate Club':        nx.karate_club_graph(),
        'Les Misérables':     nx.les_miserables_graph(),
        'Watts-Strogatz':     nx.watts_strogatz_graph(100, 6, 0.1, seed=42),
        'Barabási-Albert':    nx.barabasi_albert_graph(100, 3, seed=42),
        'Caveman':            nx.caveman_graph(10, 10),
        'Florentine Families': nx.florentine_families_graph(),
        'Davis Southern Women': nx.davis_southern_women_graph(),
        'Petersen':           nx.petersen_graph(),
        'Planted Partition':  nx.generators.community.planted_partition_graph(
                                  4, 20, 0.5, 0.1, seed=42),
    }

    # Discover available BGL binaries
    available_bgl = {n: e for n, e in ALL_BGL_VARIANTS.items()
                     if os.path.exists(e)}
    has_gen = genlouvain_available()

    all_data = []

    for graph_name, G_raw in test_graphs.items():
        G = _remap_graph(G_raw)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"\n  {graph_name} (n={n}, m={m})")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            write_edgelist(G, f.name)
            temp_file = f.name

        for trial in range(n_trials):
            seed = trial

            # ── NetworkX ──
            communities = nx.algorithms.community.louvain_communities(G, seed=seed)
            Q_nx = nx.algorithms.community.modularity(G, communities)
            all_data.append({
                'Graph': graph_name, 'Implementation': 'NetworkX',
                'Modularity': Q_nx, 'Communities': len(communities),
            })

            # ── igraph ──
            edges = list(G.edges())
            g_ig = ig.Graph(n=n, edges=edges)
            np.random.seed(seed)
            part_ig = g_ig.community_multilevel()
            all_data.append({
                'Graph': graph_name, 'Implementation': 'igraph',
                'Modularity': part_ig.modularity,
                'Communities': len(part_ig),
            })

            # ── genlouvain ──
            if has_gen:
                _, Q_gen, n_comm_gen = run_genlouvain(temp_file, seed)
                if Q_gen is not None:
                    all_data.append({
                        'Graph': graph_name, 'Implementation': 'genlouvain',
                        'Modularity': Q_gen, 'Communities': n_comm_gen,
                    })
            elif trial == 0:
                print("    (genlouvain not available)")

            # ── BGL variants ──
            for bgl_name, exe in available_bgl.items():
                _, Q, partition = run_bgl_exe(exe, temp_file, seed)
                if Q is not None:
                    all_data.append({
                        'Graph': graph_name, 'Implementation': bgl_name,
                        'Modularity': Q,
                        'Communities': len(set(partition)),
                    })

        os.unlink(temp_file)

        # summary per graph
        df_g = pd.DataFrame([d for d in all_data if d['Graph'] == graph_name])
        for impl in df_g['Implementation'].unique():
            df_i = df_g[df_g['Implementation'] == impl]
            print(f"    {impl:20s}: Q={df_i['Modularity'].mean():.4f} "
                  f"(±{df_i['Modularity'].std():.4f}), "
                  f"communities={df_i['Communities'].mean():.1f}")

    df = pd.DataFrame(all_data)
    df.to_csv('results/correctness.csv', index=False)
    print(f"\nSaved to results/correctness.csv ({len(df)} data points)")
    return df


# ── runtime scalability ─────────────────────────────────────────────────

def run_benchmark_runtime(n_trials=10, sizes=None):
    """Run runtime scalability benchmark.

    Benchmark fairness notes:
      - All C++ implementations (BGL, gen-louvain) compiled with -O3.
      - igraph is a pre-compiled pip wheel (release-optimized).
      - Timing measures ONLY the clustering algorithm, never graph I/O:
          * BGL & gen-louvain: internal std::chrono timing (LOUVAIN_TIME).
          * igraph & NetworkX: time.time() wrapping only the community call.
      - One warm-up trial per implementation is run before measured trials
        to populate CPU/file-system caches and JIT any interpreter paths.
      - The igraph graph object is built once and reused across trials.

    Output: results/runtime.csv
        Columns: GraphType, Nodes, Edges, Implementation,
                 Time, Time_Std, Communities, Communities_Std
    """
    print("\nRuntime Benchmark")
    print("=" * 60)

    if sizes is None:
        sizes = [1000, 5000, 10000, 50000, 100000, 250000]
    graph_types = ['LFR', 'ScaleFree']

    available_bgl = {n: e for n, e in ALL_BGL_VARIANTS.items()
                     if os.path.exists(e)}
    has_gen = genlouvain_available()

    results = []

    for graph_type in graph_types:
        print(f"\n{graph_type}:")
        for n in sizes:
            print(f"  n={n:,}...", end=' ', flush=True)

            if graph_type == 'LFR':
                try:
                    G = nx.generators.community.LFR_benchmark_graph(
                        n, tau1=3, tau2=1.5, mu=0.1, average_degree=5,
                        max_degree=min(50, n // 10),
                        min_community=max(10, n // 100),
                        max_community=min(n // 10, n // 2), seed=42)
                except Exception:
                    G = nx.powerlaw_cluster_graph(n, 2, 0.1, seed=42)
            else:
                G = nx.barabasi_albert_graph(n, 3, seed=42)

            m = G.number_of_edges()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                             delete=False) as f:
                write_edgelist(G, f.name)
                temp_file = f.name

            # Build igraph graph once
            edges = list(G.edges())
            g_ig = ig.Graph(n=G.number_of_nodes(), edges=edges)

            # ── warm-up (results discarded) ──
            nx.algorithms.community.louvain_communities(G, seed=42)
            g_ig.community_multilevel()
            if 'BGL vecS/vecS' in available_bgl:
                subprocess.run(
                    [available_bgl['BGL vecS/vecS'], temp_file, '42'],
                    capture_output=True, text=True, timeout=900,
                )
            if has_gen:
                run_genlouvain(temp_file, 42)

            summary_parts = []

            # ── NetworkX ──
            times, comms = [], []
            for _ in range(n_trials):
                start = time.time()
                communities = nx.algorithms.community.louvain_communities(
                    G, seed=42)
                times.append(time.time() - start)
                comms.append(len(communities))
            results.append({
                'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                'Implementation': 'NetworkX',
                'Time': np.mean(times), 'Time_Std': np.std(times),
                'Communities': np.mean(comms),
                'Communities_Std': np.std(comms),
            })
            summary_parts.append(f"NX {np.mean(times):.3f}s")

            # ── igraph ──
            times, comms = [], []
            for _ in range(n_trials):
                start = time.time()
                part = g_ig.community_multilevel()
                times.append(time.time() - start)
                comms.append(len(part))
            results.append({
                'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                'Implementation': 'igraph',
                'Time': np.mean(times), 'Time_Std': np.std(times),
                'Communities': np.mean(comms),
                'Communities_Std': np.std(comms),
            })
            summary_parts.append(f"ig {np.mean(times):.3f}s")

            # ── genlouvain ──
            if has_gen:
                times = []
                for _ in range(n_trials):
                    t_gen, _, _ = run_genlouvain(temp_file, 42)
                    if t_gen is not None:
                        times.append(t_gen)
                if times:
                    results.append({
                        'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                        'Implementation': 'genlouvain',
                        'Time': np.mean(times), 'Time_Std': np.std(times),
                        'Communities': float('nan'),
                        'Communities_Std': float('nan'),
                    })
                    summary_parts.append(f"gen {np.mean(times):.3f}s")

            # ── BGL variants ──
            for bgl_name, exe in available_bgl.items():
                if bgl_name == 'BGL adj_matrix' and n > MATRIX_MAX_NODES:
                    results.append({
                        'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                        'Implementation': bgl_name,
                        'Time': float('nan'), 'Time_Std': float('nan'),
                        'Communities': float('nan'),
                        'Communities_Std': float('nan'),
                    })
                    short = bgl_name.split()[-1]
                    summary_parts.append(f"{short} SKIP")
                    continue

                times, comms = [], []
                for _ in range(n_trials):
                    t, _, partition = run_bgl_exe(exe, temp_file)
                    if t is not None:
                        times.append(t)
                    if partition is not None:
                        comms.append(len(set(partition)))

                if times:
                    results.append({
                        'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                        'Implementation': bgl_name,
                        'Time': np.mean(times), 'Time_Std': np.std(times),
                        'Communities': np.mean(comms) if comms else float('nan'),
                        'Communities_Std': np.std(comms) if comms else float('nan'),
                    })
                    short = bgl_name.split()[-1]
                    summary_parts.append(f"{short} {np.mean(times):.4f}s")
                else:
                    results.append({
                        'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                        'Implementation': bgl_name,
                        'Time': float('nan'), 'Time_Std': float('nan'),
                        'Communities': float('nan'),
                        'Communities_Std': float('nan'),
                    })
                    short = bgl_name.split()[-1]
                    summary_parts.append(f"{short} FAIL")

            os.unlink(temp_file)
            print(', '.join(summary_parts))

    df = pd.DataFrame(results)
    df.to_csv('results/runtime.csv', index=False)
    print(f"\nSaved to results/runtime.csv")
    return df


# ── incremental vs non-incremental quality function ─────────────────────

def run_benchmark_incremental(n_trials=10, sizes=None):
    """Compare incremental vs non-incremental quality function.

    The incremental path computes gain() in O(degree) per candidate move;
    the non-incremental path recomputes full graph modularity in O(E).
    Non-incremental is dramatically slower, so sizes are kept small.

    For each BGL graph-type variant, we run both the incremental (default)
    and non-incremental executable on the same graphs and measure:
      - runtime (Time, Time_Std)
      - correctness (Modularity, Communities)

    Output: results/incremental.csv
        Columns: GraphType, Nodes, Edges, Variant, Mode, Time, Time_Std,
                 Modularity, Modularity_Std, Communities, Communities_Std
    """
    print("\nIncremental vs Non-Incremental Quality Function Benchmark")
    print("=" * 60)

    if sizes is None:
        sizes = [100, 500, 1000]
    graph_types = ['LFR', 'ScaleFree']

    available_inc = {n: e for n, e in ALL_BGL_VARIANTS.items()
                     if os.path.exists(e)}
    available_noinc = {n: e for n, e in ALL_BGL_VARIANTS_NOINC.items()
                       if os.path.exists(e)}

    results = []

    for graph_type in graph_types:
        print(f"\n{graph_type}:")
        for n in sizes:
            print(f"  n={n:,}...", flush=True)

            if graph_type == 'LFR':
                try:
                    G = nx.generators.community.LFR_benchmark_graph(
                        n, tau1=3, tau2=1.5, mu=0.1, average_degree=5,
                        max_degree=min(50, n // 10),
                        min_community=max(10, n // 100),
                        max_community=min(n // 10, n // 2), seed=42)
                except Exception:
                    G = nx.powerlaw_cluster_graph(n, 2, 0.1, seed=42)
            else:
                G = nx.barabasi_albert_graph(n, 3, seed=42)

            m = G.number_of_edges()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                             delete=False) as f:
                write_edgelist(G, f.name)
                temp_file = f.name

            # ── Incremental variants ──
            for bgl_name, exe in available_inc.items():
                variant = bgl_name  # e.g. "BGL vecS/vecS"
                if 'adj_matrix' in bgl_name and n > MATRIX_MAX_NODES:
                    results.append({
                        'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                        'Variant': variant, 'Mode': 'incremental',
                        'Time': float('nan'), 'Time_Std': float('nan'),
                        'Modularity': float('nan'), 'Modularity_Std': float('nan'),
                        'Communities': float('nan'), 'Communities_Std': float('nan'),
                    })
                    continue

                times, mods, comms = [], [], []
                for _ in range(n_trials):
                    t, Q, partition = run_bgl_exe(exe, temp_file, timeout=300)
                    if t is not None:
                        times.append(t)
                    if Q is not None:
                        mods.append(Q)
                    if partition is not None:
                        comms.append(len(set(partition)))

                results.append({
                    'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                    'Variant': variant, 'Mode': 'incremental',
                    'Time': np.mean(times) if times else float('nan'),
                    'Time_Std': np.std(times) if times else float('nan'),
                    'Modularity': np.mean(mods) if mods else float('nan'),
                    'Modularity_Std': np.std(mods) if mods else float('nan'),
                    'Communities': np.mean(comms) if comms else float('nan'),
                    'Communities_Std': np.std(comms) if comms else float('nan'),
                })
                if times:
                    print(f"    {variant:30s} inc  {np.mean(times):.6f}s  "
                          f"Q={np.mean(mods):.4f}")

            # ── Non-incremental variants ──
            for bgl_name_ni, exe_ni in available_noinc.items():
                variant = NOINC_TO_INC[bgl_name_ni]  # map back to base name
                if 'adj_matrix' in bgl_name_ni and n > MATRIX_MAX_NODES:
                    results.append({
                        'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                        'Variant': variant, 'Mode': 'non-incremental',
                        'Time': float('nan'), 'Time_Std': float('nan'),
                        'Modularity': float('nan'), 'Modularity_Std': float('nan'),
                        'Communities': float('nan'), 'Communities_Std': float('nan'),
                    })
                    continue

                # Non-incremental is very slow — use generous timeout
                times, mods, comms = [], [], []
                for _ in range(n_trials):
                    t, Q, partition = run_bgl_exe(exe_ni, temp_file, timeout=600)
                    if t is not None:
                        times.append(t)
                    if Q is not None:
                        mods.append(Q)
                    if partition is not None:
                        comms.append(len(set(partition)))

                results.append({
                    'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                    'Variant': variant, 'Mode': 'non-incremental',
                    'Time': np.mean(times) if times else float('nan'),
                    'Time_Std': np.std(times) if times else float('nan'),
                    'Modularity': np.mean(mods) if mods else float('nan'),
                    'Modularity_Std': np.std(mods) if mods else float('nan'),
                    'Communities': np.mean(comms) if comms else float('nan'),
                    'Communities_Std': np.std(comms) if comms else float('nan'),
                })
                if times:
                    print(f"    {variant:30s} noinc {np.mean(times):.6f}s  "
                          f"Q={np.mean(mods):.4f}")
                else:
                    print(f"    {variant:30s} noinc TIMEOUT/FAIL")

            os.unlink(temp_file)

    df = pd.DataFrame(results)
    df.to_csv('results/incremental.csv', index=False)
    print(f"\nSaved to results/incremental.csv")
    return df


# ── main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Louvain benchmark suite')
    parser.add_argument('--quick', action='store_true',
                        help='Run a fast smoke-test (fewer graphs, trials, sizes)')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    if args.quick:
        print('*** QUICK MODE — reduced trials & sizes ***\n')
        run_benchmark_correctness(n_trials=5)
        run_benchmark_runtime(n_trials=3,
                              sizes=[1000, 5000, 25000])
        run_benchmark_incremental(n_trials=3,
                                  sizes=[100, 500])
    else:
        run_benchmark_correctness()
        run_benchmark_runtime()
        run_benchmark_incremental()
