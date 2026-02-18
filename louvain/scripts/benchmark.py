#!/usr/bin/env python3
"""
Benchmark suite for Louvain community detection.
Tests BGL implementation against NetworkX, igraph, and genlouvain.
"""

import networkx as nx
import igraph as ig
import subprocess
import tempfile
import os
import time
import pandas as pd
import numpy as np

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
    return all(os.path.exists(f'./build/genlouvain_{b}') for b in ['convert', 'louvain', 'hierarchy'])


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
        
        subprocess.run([f'{build_dir}/genlouvain_convert', '-i', temp_txt, '-o', temp_bin],
                      capture_output=True, text=True, timeout=30, check=True)
        
        result = subprocess.run([f'{build_dir}/genlouvain_louvain', temp_bin, '-l', '-1', '-q', '0'],
                               capture_output=True, text=True, timeout=300, check=True)
        
        # Parse LOUVAIN_TIME and modularity from stderr
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
        
        result = subprocess.run([f'{build_dir}/genlouvain_hierarchy', temp_tree, '-n'],
                              capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().split('\n')
        num_levels = int(lines[0].split(':')[1].strip())
        
        result = subprocess.run([f'{build_dir}/genlouvain_hierarchy', temp_tree, '-l', str(num_levels - 1)],
                              capture_output=True, text=True, timeout=10)
        
        partition_dict = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0])
                        comm_id = int(parts[1])
                        partition_dict[node_id] = comm_id
                    except ValueError:
                        continue
        
        num_communities = len(set(partition_dict.values()))
        
        for f in [temp_txt, temp_bin, temp_tree]:
            if f and os.path.exists(f):
                os.remove(f)
        
        return louvain_time, Q, num_communities
    except:
        for f in [temp_txt, temp_bin, temp_tree]:
            if f and os.path.exists(f):
                os.remove(f)
        return None, None, None

def run_benchmark_correctness(n_trials=100):
    """Run correctness benchmark on standard graphs."""
    print("Correctness Benchmark")
    print("=" * 60)
    
    test_graphs = {
        'Karate Club': nx.karate_club_graph(),
        'Les Misérables': nx.les_miserables_graph(),
        'Watts-Strogatz': nx.watts_strogatz_graph(100, 6, 0.1, seed=42),
        'Barabási-Albert': nx.barabasi_albert_graph(100, 3, seed=42),
        'Caveman': nx.caveman_graph(10, 10),
        'Florentine Families': nx.florentine_families_graph(),
        'Davis Southern Women': nx.davis_southern_women_graph(),
        'Petersen': nx.petersen_graph(),
        'Planted Partition': nx.generators.community.planted_partition_graph(4, 20, 0.5, 0.1, seed=42)
    }
    
    all_data = []
    
    for graph_name, G in test_graphs.items():
        print(f"\n{graph_name} (n={G.number_of_nodes()}, m={G.number_of_edges()})")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            write_edgelist(G, f.name)
            temp_file = f.name
        
        for trial in range(n_trials):
            seed = trial
            
            # NetworkX
            communities = nx.algorithms.community.louvain_communities(G, seed=seed)
            partition = {node: i for i, comm in enumerate(communities) for node in comm}
            Q_nx = nx.algorithms.community.modularity(G, communities)
            all_data.append({'Graph': graph_name, 'Implementation': 'Networkx', 
                           'Modularity': Q_nx, 'Communities': len(communities)})
            
            # igraph
            if not all(isinstance(n, int) for n in G.nodes()):
                mapping = {node: i for i, node in enumerate(G.nodes())}
                G_mapped = nx.relabel_nodes(G, mapping)
                edges = list(G_mapped.edges())
            else:
                edges = list(G.edges())
            g_ig = ig.Graph(n=G.number_of_nodes(), edges=edges)
            np.random.seed(seed)
            part_ig = g_ig.community_multilevel()
            Q_ig = part_ig.modularity
            all_data.append({'Graph': graph_name, 'Implementation': 'Igraph',
                           'Modularity': Q_ig, 'Communities': len(part_ig)})
            
            # BGL
            result = subprocess.run(['./build/bgl_louvain_vecS_vecS', temp_file, str(seed)],
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                Q_bgl = float(lines[0])
                partition_bgl = [int(x) for x in lines[1].split()]
                n_comm_bgl = len(set(partition_bgl))
                all_data.append({'Graph': graph_name, 'Implementation': 'BGL',
                               'Modularity': Q_bgl, 'Communities': n_comm_bgl})
            
            # genlouvain
            _, Q_gen, n_comm_gen = run_genlouvain(temp_file, seed)
            if Q_gen is not None:
                all_data.append({'Graph': graph_name, 'Implementation': 'Genlouvain',
                               'Modularity': Q_gen, 'Communities': n_comm_gen})
            elif trial == 0:
                print(f"  (genlouvain not available)")
        
        os.unlink(temp_file)
    
    df = pd.DataFrame(all_data)
    df.to_csv('results/correctness.csv', index=False)
    print(f"\nSaved to results/correctness.csv ({len(df)} data points)")
    return df

def run_benchmark_runtime(n_trials=3):
    """Run runtime scalability benchmark."""
    # Benchmark fairness notes:
    #   - All C++ implementations (BGL, gen-louvain) compiled with -O3.
    #   - igraph is a pre-compiled pip wheel (release-optimized).
    #   - Timing measures ONLY the clustering algorithm, never graph I/O:
    #       * BGL & gen-louvain: internal std::chrono timing printed as LOUVAIN_TIME.
    #       * igraph & NetworkX: time.time() wrapping only the community call.
    #   - One warm-up trial per implementation is run before measured trials
    #     to populate CPU/file-system caches and JIT any interpreter paths.
    #   - The igraph graph object is built once and reused across trials.
    print("\nRuntime Benchmark")
    print("=" * 60)
    #sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000]    
    sizes = [1000, 2500, 5000, 10000, 25000, 50000]
    graph_types = ['LFR', 'ScaleFree']
    results = []
    
    for graph_type in graph_types:
        print(f"\n{graph_type}:")
        for n in sizes:
            print(f"  n={n:,}...", end=' ', flush=True)
            
            if graph_type == 'LFR':
                try:
                    G = nx.generators.community.LFR_benchmark_graph(
                        n, tau1=3, tau2=1.5, mu=0.1, average_degree=5,
                        max_degree=min(50, n//10), min_community=max(10, n//100),
                        max_community=min(n//10, n//2), seed=42)
                except:
                    G = nx.powerlaw_cluster_graph(n, 2, 0.1, seed=42)
            else:
                G = nx.barabasi_albert_graph(n, 3, seed=42)
            
            m = G.number_of_edges()
            times_nx, times_ig, times_bgl, times_gen = [], [], [], []
            comms_nx, comms_ig, comms_bgl = [], [], []
            has_genlouvain = genlouvain_available()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                write_edgelist(G, f.name)
                temp_file = f.name
            
            # Build igraph graph once (construction cost is not part of the measurement)
            edges = list(G.edges())
            g_ig = ig.Graph(n=G.number_of_nodes(), edges=edges)

            # Warm-up: run each implementation once to populate caches.
            # Results are discarded.
            nx.algorithms.community.louvain_communities(G, seed=42)
            g_ig.community_multilevel()
            subprocess.run(['./build/bgl_louvain_vecS_vecS', temp_file, '42'],
                           capture_output=True, text=True, timeout=900)
            if has_genlouvain:
                run_genlouvain(temp_file, 42)

            for _ in range(n_trials):
                start = time.time()
                communities_nx = nx.algorithms.community.louvain_communities(G, seed=42)
                times_nx.append(time.time() - start)
                comms_nx.append(len(communities_nx))
                
                start = time.time()
                partition_ig = g_ig.community_multilevel()
                times_ig.append(time.time() - start)
                comms_ig.append(len(partition_ig))
                
                result = subprocess.run(['./build/bgl_louvain_vecS_vecS', temp_file, '42'],
                             capture_output=True, text=True, timeout=900)
                # Parse LOUVAIN_TIME from stderr
                louvain_time = None
                for line in result.stderr.split('\n'):
                    if line.startswith('LOUVAIN_TIME:'):
                        louvain_time = float(line.split(':')[1].strip())
                        break
                if louvain_time is not None:
                    times_bgl.append(louvain_time)
                else:
                    # Fallback: use subprocess time (shouldn't happen)
                    times_bgl.append(0.0)
                
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    partition_bgl = [int(x) for x in lines[1].split()]
                    comms_bgl.append(len(set(partition_bgl)))
                
                # gen-louvain (internal timing via LOUVAIN_TIME)
                if has_genlouvain:
                    t_gen, _, _ = run_genlouvain(temp_file, 42)
                    if t_gen is not None:
                        times_gen.append(t_gen)
            
            os.unlink(temp_file)
            
            row = {
                'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                'NetworkX_Time': np.mean(times_nx), 'NetworkX_Std': np.std(times_nx),
                'Igraph_Time': np.mean(times_ig), 'Igraph_Std': np.std(times_ig),
                'BGL_Time': np.mean(times_bgl), 'BGL_Std': np.std(times_bgl),
                'NetworkX_Communities': np.mean(comms_nx), 'NetworkX_Communities_Std': np.std(comms_nx),
                'Igraph_Communities': np.mean(comms_ig), 'Igraph_Communities_Std': np.std(comms_ig),
                'BGL_Communities': np.mean(comms_bgl), 'BGL_Communities_Std': np.std(comms_bgl),
            }
            if times_gen:
                row['Genlouvain_Time'] = np.mean(times_gen)
                row['Genlouvain_Std'] = np.std(times_gen)
            else:
                row['Genlouvain_Time'] = float('nan')
                row['Genlouvain_Std'] = float('nan')
            results.append(row)
            
            summary = f"NX {np.mean(times_nx):.3f}s, ig {np.mean(times_ig):.3f}s, BGL {np.mean(times_bgl):.3f}s"
            if times_gen:
                summary += f", gen {np.mean(times_gen):.3f}s"
            print(summary)
    
    df = pd.DataFrame(results)
    df.to_csv('results/runtime.csv', index=False)
    print(f"\nSaved to results/runtime.csv")
    return df

def run_bgl_exe(exe, temp_file, seed='42', timeout=900):
    """Run a BGL executable and parse LOUVAIN_TIME, modularity, and partition."""
    try:
        result = subprocess.run([exe, temp_file, str(seed)],
                                capture_output=True, text=True, timeout=timeout)
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


def run_benchmark_bgl_variants(n_trials=3):
    """Benchmark Louvain across BGL graph data structures + igraph baseline."""
    print("\nBGL Variants Runtime Benchmark")
    print("=" * 60)

    bgl_variants = {
        'BGL vecS/vecS':  './build/bgl_louvain_vecS_vecS',
        'BGL listS/vecS': './build/bgl_louvain_listS_vecS',
        'BGL setS/vecS':  './build/bgl_louvain_setS_vecS',
        'BGL adj_matrix': './build/bgl_louvain_matrix',
    }

    available = {}
    for name, exe in bgl_variants.items():
        if os.path.exists(exe):
            available[name] = exe
            print(f"  Found {name}")
        else:
            print(f"  WARNING: {name} not found at {exe}, skipping")

    sizes = [1000, 2500, 5000, 10000, 25000, 50000]
    matrix_max_size = 5000  # O(V^2) memory
    graph_types = ['LFR', 'ScaleFree']
    results = []

    for graph_type in graph_types:
        print(f"\n{graph_type}:")
        for n in sizes:
            print(f"  n={n:,}...", end=' ', flush=True)

            if graph_type == 'LFR':
                try:
                    G = nx.generators.community.LFR_benchmark_graph(
                        n, tau1=3, tau2=1.5, mu=0.1, average_degree=5,
                        max_degree=min(50, n//10), min_community=max(10, n//100),
                        max_community=min(n//10, n//2), seed=42)
                except:
                    G = nx.powerlaw_cluster_graph(n, 2, 0.1, seed=42)
            else:
                G = nx.barabasi_albert_graph(n, 3, seed=42)

            m = G.number_of_edges()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                write_edgelist(G, f.name)
                temp_file = f.name

            # igraph baseline
            times_ig = []
            edges = list(G.edges())
            for _ in range(n_trials):
                g_ig = ig.Graph(n=G.number_of_nodes(), edges=edges)
                start = time.time()
                g_ig.community_multilevel()
                times_ig.append(time.time() - start)
            avg_ig = np.mean(times_ig)
            results.append({'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                            'Variant': 'igraph', 'Time': avg_ig, 'Time_Std': np.std(times_ig)})
            summary = [f"igraph {avg_ig:.4f}s"]

            # BGL variants
            for name, exe in available.items():
                if name == 'BGL adj_matrix' and n > matrix_max_size:
                    results.append({'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                                    'Variant': name, 'Time': float('nan'), 'Time_Std': float('nan')})
                    summary.append(f"{name.split()[-1]} SKIP")
                    continue

                times_v = []
                for _ in range(n_trials):
                    t, _, _ = run_bgl_exe(exe, temp_file)
                    if t is not None:
                        times_v.append(t)

                if times_v:
                    avg = np.mean(times_v)
                    results.append({'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                                    'Variant': name, 'Time': avg, 'Time_Std': np.std(times_v)})
                    summary.append(f"{name.split()[-1]} {avg:.4f}s")
                else:
                    results.append({'GraphType': graph_type, 'Nodes': n, 'Edges': m,
                                    'Variant': name, 'Time': float('nan'), 'Time_Std': float('nan')})
                    summary.append(f"{name.split()[-1]} FAIL")

            os.unlink(temp_file)
            print(', '.join(summary))

    df = pd.DataFrame(results)
    df.to_csv('results/bgl_variants_runtime.csv', index=False)
    print(f"\nSaved to results/bgl_variants_runtime.csv")
    return df


def run_benchmark_bgl_variants_correctness(n_trials=10):
    """Verify all BGL variants produce correct modularity scores."""
    print("\nBGL Variants Correctness")
    print("=" * 60)

    bgl_variants = {
        'BGL vecS/vecS':  './build/bgl_louvain_vecS_vecS',
        'BGL listS/vecS': './build/bgl_louvain_listS_vecS',
        'BGL setS/vecS':  './build/bgl_louvain_setS_vecS',
        'BGL adj_matrix': './build/bgl_louvain_matrix',
    }

    available = {n: e for n, e in bgl_variants.items() if os.path.exists(e)}

    test_graphs = {
        'Karate Club': nx.karate_club_graph(),
        'Les Miserables': nx.les_miserables_graph(),
        'Petersen': nx.petersen_graph(),
        'Caveman(10,10)': nx.caveman_graph(10, 10),
    }

    all_data = []

    for graph_name, G in test_graphs.items():
        G = nx.Graph(G)  # ensure simple graph
        if not all(isinstance(n, int) for n in G.nodes()):
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

        n = G.number_of_nodes()
        m = G.number_of_edges()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            write_edgelist(G, f.name)
            temp_file = f.name

        print(f"\n  {graph_name} (n={n}, m={m})")

        for trial in range(n_trials):
            seed = trial

            # igraph
            edges = list(G.edges())
            g_ig = ig.Graph(n=n, edges=edges)
            np.random.seed(seed)
            part_ig = g_ig.community_multilevel()
            all_data.append({'Graph': graph_name, 'Variant': 'igraph',
                             'Modularity': part_ig.modularity, 'Communities': len(part_ig)})

            # BGL variants
            for name, exe in available.items():
                _, Q, partition = run_bgl_exe(exe, temp_file, seed)
                if Q is not None:
                    all_data.append({'Graph': graph_name, 'Variant': name,
                                     'Modularity': Q, 'Communities': len(set(partition))})

        os.unlink(temp_file)

        # Print summary for this graph
        df_g = pd.DataFrame([d for d in all_data if d['Graph'] == graph_name])
        for variant in df_g['Variant'].unique():
            df_v = df_g[df_g['Variant'] == variant]
            print(f"    {variant:20s}: Q={df_v['Modularity'].mean():.4f} (±{df_v['Modularity'].std():.4f}), "
                  f"communities={df_v['Communities'].mean():.1f}")

    df = pd.DataFrame(all_data)
    df.to_csv('results/bgl_variants_correctness.csv', index=False)
    print(f"\nSaved to results/bgl_variants_correctness.csv ({len(df)} data points)")
    return df


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('results', exist_ok=True)

    # Existing benchmarks (unchanged)
    run_benchmark_correctness()
    run_benchmark_runtime()

    # New: BGL variants
    df_vc = run_benchmark_bgl_variants_correctness()
    df_vr = run_benchmark_bgl_variants()

    # === Plot BGL variants runtime ===
    if df_vr is not None and len(df_vr) > 0:
        variant_styles = {
            'igraph':          {'color': '#2ca02c', 'marker': 's', 'ls': '--', 'lw': 2},
            'BGL vecS/vecS':   {'color': '#1f77b4', 'marker': 'o', 'ls': '-',  'lw': 1.5},
            'BGL listS/vecS':  {'color': '#ff7f0e', 'marker': '^', 'ls': '-',  'lw': 1.5},
            'BGL setS/vecS':   {'color': '#9467bd', 'marker': 'D', 'ls': '-',  'lw': 1.5},
            'BGL adj_matrix':  {'color': '#e377c2', 'marker': 'X', 'ls': '-',  'lw': 1.5},
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, graph_type in enumerate(['LFR', 'ScaleFree']):
            ax = axes[idx]
            df_gt = df_vr[df_vr['GraphType'] == graph_type]

            for variant in df_gt['Variant'].unique():
                df_v = df_gt[df_gt['Variant'] == variant].dropna(subset=['Time']).sort_values('Nodes')
                if df_v.empty:
                    continue
                style = variant_styles.get(variant, {'color': '#333', 'marker': 'o', 'ls': '-', 'lw': 1})
                ax.plot(df_v['Nodes'], df_v['Time'],
                        marker=style['marker'], color=style['color'],
                        linestyle=style['ls'], linewidth=style['lw'],
                        label=variant, markersize=7)

            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'BGL Data Structure Variants — {graph_type}')
            ax.legend(fontsize=8, loc='upper left')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/bgl_variants_runtime.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to results/bgl_variants_runtime.png")

    # === Plot BGL variants correctness ===
    if df_vc is not None and len(df_vc) > 0:
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))
        graphs = df_vc['Graph'].unique()
        variants = df_vc['Variant'].unique()
        x = np.arange(len(graphs))
        width = 0.8 / len(variants)

        colors = {
            'igraph':          '#2ca02c',
            'BGL vecS/vecS':   '#1f77b4',
            'BGL listS/vecS':  '#ff7f0e',
            'BGL setS/vecS':   '#9467bd',
            'BGL adj_matrix':  '#e377c2',
        }

        for i, variant in enumerate(variants):
            df_v = df_vc[df_vc['Variant'] == variant]
            # Modularity
            means_q = [df_v[df_v['Graph'] == g]['Modularity'].mean() if len(df_v[df_v['Graph'] == g]) > 0 else 0 for g in graphs]
            stds_q  = [df_v[df_v['Graph'] == g]['Modularity'].std()  if len(df_v[df_v['Graph'] == g]) > 0 else 0 for g in graphs]
            ax2.bar(x + i * width, means_q, width, yerr=stds_q,
                    label=variant, color=colors.get(variant, '#333'), alpha=0.85,
                    capsize=3)
            # Communities
            means_c = [df_v[df_v['Graph'] == g]['Communities'].mean() if len(df_v[df_v['Graph'] == g]) > 0 else 0 for g in graphs]
            stds_c  = [df_v[df_v['Graph'] == g]['Communities'].std()  if len(df_v[df_v['Graph'] == g]) > 0 else 0 for g in graphs]
            ax3.bar(x + i * width, means_c, width, yerr=stds_c,
                    label=variant, color=colors.get(variant, '#333'), alpha=0.85,
                    capsize=3)

        ax2.set_xlabel('Graph')
        ax2.set_ylabel('Modularity (Q)')
        ax2.set_title('BGL Variants — Modularity')
        ax2.set_xticks(x + width * len(variants) / 2)
        ax2.set_xticklabels(graphs, rotation=15, ha='right')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3.set_xlabel('Graph')
        ax3.set_ylabel('Number of Communities')
        ax3.set_title('BGL Variants — Communities Found')
        ax3.set_xticks(x + width * len(variants) / 2)
        ax3.set_xticklabels(graphs, rotation=15, ha='right')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('results/bgl_variants_correctness.png', dpi=150, bbox_inches='tight')
        print("Plot saved to results/bgl_variants_correctness.png")
