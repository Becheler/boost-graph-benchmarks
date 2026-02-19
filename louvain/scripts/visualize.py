#!/usr/bin/env python3
"""Visualize Louvain benchmark results — one PNG per plot.

Reads two unified CSVs produced by benchmark.py:
  results/correctness.csv  — per-trial rows for every implementation
  results/runtime.csv      — aggregated rows (long format) for every impl
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Canonical ordering and colours for all implementations.
ALL_IMPLEMENTATIONS = [
    'NetworkX', 'igraph', 'genlouvain',
    'BGL vecS/vecS', 'BGL listS/vecS', 'BGL setS/vecS', 'BGL adj_matrix',
]
IMPL_COLORS = {
    'NetworkX':        '#1f77b4',
    'igraph':          '#ff7f0e',
    'genlouvain':      '#d62728',
    'BGL vecS/vecS':   '#2ca02c',
    'BGL listS/vecS':  '#9467bd',
    'BGL setS/vecS':   '#8c564b',
    'BGL adj_matrix':  '#e377c2',
}

# Subsets used by specific plot families
REFS = ['NetworkX', 'igraph', 'genlouvain']
BGL_VARIANTS = ['BGL vecS/vecS', 'BGL listS/vecS', 'BGL setS/vecS', 'BGL adj_matrix']

os.makedirs('results', exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────


def _ordered(impls, df_col):
    """Return impls that actually appear in *df_col*, in canonical order."""
    present = set(df_col.unique())
    return [i for i in impls if i in present]


def _save(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close(fig)


# ── correctness plots ────────────────────────────────────────────────────


def _boxplot_by_graph(df, column, xlabel, title, filename, impls=None):
    """Horizontal boxplots of *column* grouped by Graph × Implementation."""
    if impls is None:
        impls = _ordered(ALL_IMPLEMENTATIONS, df['Implementation'])
    else:
        impls = _ordered(impls, df['Implementation'])

    graphs = df['Graph'].unique()
    fig, ax = plt.subplots(figsize=(10, max(4, len(graphs) * 1.4)))

    positions, data, box_colors = [], [], []
    for i, graph in enumerate(graphs):
        for j, impl in enumerate(impls):
            subset = df[(df['Graph'] == graph) & (df['Implementation'] == impl)]
            vals = subset[column].dropna().values if len(subset) > 0 else np.array([])
            if len(vals) > 0:
                data.append(vals)
                positions.append(i * (len(impls) + 1) + j)
                box_colors.append(IMPL_COLORS.get(impl, '#333333'))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(linewidth=1.5), vert=False)
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c); patch.set_edgecolor(c); patch.set_alpha(0.7)
    for k, c in enumerate(box_colors):
        bp['whiskers'][k * 2].set_color(c); bp['whiskers'][k * 2 + 1].set_color(c)
        bp['caps'][k * 2].set_color(c); bp['caps'][k * 2 + 1].set_color(c)
        bp['medians'][k].set_color(c); bp['medians'][k].set_linewidth(2)

    mid = (len(impls) - 1) / 2
    graph_positions = [i * (len(impls) + 1) + mid for i in range(len(graphs))]
    ax.set_yticks(graph_positions)
    ax.set_yticklabels(graphs, fontsize=9)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=IMPL_COLORS.get(impl, '#333'), alpha=0.7, label=impl)
        for impl in impls
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)
    plt.tight_layout()
    _save(fig, filename)


def _relative_bar(df, column, xlabel, title, filename, impls=None):
    """Horizontal bar chart of relative deviation from cross-impl mean."""
    if impls is None:
        impls = _ordered(ALL_IMPLEMENTATIONS, df['Implementation'])
    else:
        impls = _ordered(impls, df['Implementation'])

    df_mean = df.groupby(['Graph', 'Implementation'])[column].mean().reset_index()
    relative = []
    for graph, group in df_mean.groupby('Graph'):
        mean_val = group[column].mean()
        for _, row in group.iterrows():
            rel = ((row[column] - mean_val) / mean_val) * 100 if mean_val != 0 else 0
            relative.append({'Graph': graph, 'Implementation': row['Implementation'], 'Relative': rel})
    df_rel = pd.DataFrame(relative)
    pivot = df_rel.pivot(index='Graph', columns='Implementation', values='Relative')
    avail = [impl for impl in impls if impl in pivot.columns]
    pivot = pivot[avail]
    impl_colors = [IMPL_COLORS.get(impl, '#333') for impl in avail]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
    pivot.plot(kind='barh', ax=ax, color=impl_colors, width=0.75)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, filename)


def _bgl_vs_refs(df, filename):
    """Horizontal bars: each BGL variant's modularity %diff vs each reference."""
    df_mean = df.groupby(['Graph', 'Implementation'])['Modularity'].mean().reset_index()
    comparisons = []
    for graph in df_mean['Graph'].unique():
        for bgl in BGL_VARIANTS:
            bgl_q = df_mean[(df_mean['Graph'] == graph) & (df_mean['Implementation'] == bgl)]['Modularity'].values
            if len(bgl_q) == 0:
                continue
            bgl_q = bgl_q[0]
            for ref in REFS:
                ref_q = df_mean[(df_mean['Graph'] == graph) & (df_mean['Implementation'] == ref)]['Modularity'].values
                if len(ref_q) > 0 and ref_q[0] != 0:
                    diff = ((bgl_q - ref_q[0]) / ref_q[0]) * 100
                    comparisons.append({'Graph': graph, 'BGL variant': bgl, 'vs': f'vs {ref}', 'Difference': diff})
    if not comparisons:
        return
    df_comp = pd.DataFrame(comparisons)

    bgl_present = _ordered(BGL_VARIANTS, df_comp['BGL variant'])
    n_variants = len(bgl_present)
    fig, axes = plt.subplots(1, n_variants, figsize=(6 * n_variants, max(4, len(df_comp['Graph'].unique()) * 0.6)),
                             sharey=True, squeeze=False)

    ref_colors = {f'vs {r}': IMPL_COLORS[r] for r in REFS}
    for idx, bgl in enumerate(bgl_present):
        ax = axes[0][idx]
        sub = df_comp[df_comp['BGL variant'] == bgl]
        pivot = sub.pivot(index='Graph', columns='vs', values='Difference')
        col_order = [c for c in ['vs NetworkX', 'vs igraph', 'vs genlouvain'] if c in pivot.columns]
        pivot = pivot[col_order]
        bar_colors = [ref_colors.get(c, '#333') for c in col_order]
        pivot.plot(kind='barh', ax=ax, color=bar_colors, width=0.75)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('((BGL Q - Ref Q) / Ref Q) x 100', fontweight='bold')
        ax.set_title(bgl, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle('BGL Variants vs Reference Implementations (Modularity)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


# ── runtime plots ────────────────────────────────────────────────────────


def _runtime_scalability(df_rt, filename, impls=None):
    """Log-log runtime vs nodes, side-by-side panels for each graph type."""
    if impls is None:
        impls = _ordered(ALL_IMPLEMENTATIONS, df_rt['Implementation'])
    else:
        impls = _ordered(impls, df_rt['Implementation'])

    graph_types = [gt for gt in ['LFR', 'ScaleFree'] if gt in df_rt['GraphType'].values]
    fig, axes = plt.subplots(1, len(graph_types), figsize=(7 * len(graph_types), 5),
                             sharey=True, squeeze=False)

    for idx, gt in enumerate(graph_types):
        ax = axes[0][idx]
        dt = df_rt[df_rt['GraphType'] == gt]
        for i, impl in enumerate(impls):
            di = dt[dt['Implementation'] == impl].dropna(subset=['Time']).sort_values('Nodes')
            if di.empty:
                continue
            ax.errorbar(di['Nodes'], di['Time'], yerr=di['Time_Std'],
                        marker='|', markersize=6, label=impl, capsize=3,
                        linewidth=1, color=IMPL_COLORS.get(impl, '#333'))
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_title(f'{gt} Graphs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log'); ax.set_yscale('log')

    axes[0][0].set_ylabel('Time (seconds)', fontweight='bold')
    fig.suptitle('Runtime Scalability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


def _communities_detected(df_rt, filename, impls=None):
    """Communities detected vs nodes, side-by-side panels for each graph type."""
    if impls is None:
        impls = _ordered(ALL_IMPLEMENTATIONS, df_rt['Implementation'])
    else:
        impls = _ordered(impls, df_rt['Implementation'])
    if 'Communities' not in df_rt.columns:
        return

    graph_types = [gt for gt in ['LFR', 'ScaleFree'] if gt in df_rt['GraphType'].values]
    fig, axes = plt.subplots(1, len(graph_types), figsize=(7 * len(graph_types), 5),
                             sharey=True, squeeze=False)

    for idx, gt in enumerate(graph_types):
        ax = axes[0][idx]
        dt = df_rt[df_rt['GraphType'] == gt]
        for i, impl in enumerate(impls):
            di = dt[dt['Implementation'] == impl].dropna(subset=['Communities']).sort_values('Nodes')
            if di.empty:
                continue
            ax.errorbar(di['Nodes'], di['Communities'], yerr=di['Communities_Std'],
                        marker='|', markersize=6, label=impl, capsize=3,
                        linewidth=1, color=IMPL_COLORS.get(impl, '#333'))
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_title(f'{gt} Graphs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    axes[0][0].set_ylabel('Communities Found', fontweight='bold')
    fig.suptitle('Communities Detected vs Graph Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


def _speedup_over_igraph(df_rt, filename):
    """Speedup of every other implementation relative to igraph."""
    others = [i for i in ALL_IMPLEMENTATIONS if i != 'igraph']
    others = _ordered(others, df_rt['Implementation'])
    if not others:
        return

    graph_types = [gt for gt in ['LFR', 'ScaleFree'] if gt in df_rt['GraphType'].values]
    fig, axes = plt.subplots(1, len(graph_types), figsize=(7 * len(graph_types), 5), sharey=True, squeeze=False)

    for idx, gt in enumerate(graph_types):
        ax = axes[0][idx]
        dt = df_rt[df_rt['GraphType'] == gt]
        ig = dt[dt['Implementation'] == 'igraph'][['Nodes', 'Time']].rename(columns={'Time': 'ig_time'})
        for j, impl in enumerate(others):
            di = dt[dt['Implementation'] == impl][['Nodes', 'Time']].rename(columns={'Time': 'impl_time'})
            merged = pd.merge(ig, di, on='Nodes').dropna()
            if merged.empty:
                continue
            ax.plot(merged['Nodes'], merged['ig_time'] / merged['impl_time'],
                    marker='|', markersize=6, linewidth=1,
                    color=IMPL_COLORS.get(impl, '#333'), label=impl)
        ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xscale('log')
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_title(f'{gt} Graphs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel('Speedup (igraph time / impl time)', fontweight='bold')
    fig.suptitle('Speedup over igraph', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


# ── main ─────────────────────────────────────────────────────────────────

# Correctness
if os.path.exists('results/correctness.csv'):
    df = pd.read_csv('results/correctness.csv')
    print(f"Loaded {len(df)} data points from results/correctness.csv")
    print(f"Graphs: {', '.join(df['Graph'].unique())}")
    print(f"Implementations: {', '.join(df['Implementation'].unique())}")
    print()

    _boxplot_by_graph(df, 'Modularity', 'Modularity (Q)',
                      'Modularity per Graph', 'results/correctness_modularity.png')
    _boxplot_by_graph(df, 'Communities', 'Number of Communities',
                      'Communities Found per Graph', 'results/correctness_communities.png')
    _relative_bar(df, 'Modularity',
                  '((Impl Q - Mean Q) / Mean Q) x 100',
                  'Relative Modularity Performance', 'results/relative_modularity.png')
    _relative_bar(df, 'Communities',
                  '((Impl Comms - Mean Comms) / Mean Comms) x 100',
                  'Relative Communities Performance', 'results/relative_communities.png')
    _bgl_vs_refs(df, 'results/bgl_vs_refs.png')

# Runtime
if os.path.exists('results/runtime.csv'):
    df_rt = pd.read_csv('results/runtime.csv')
    print(f"Loaded {len(df_rt)} rows from results/runtime.csv")
    print(f"Implementations: {', '.join(df_rt['Implementation'].unique())}")
    print()
    _runtime_scalability(df_rt, 'results/runtime.png')
    _communities_detected(df_rt, 'results/communities.png')
    _speedup_over_igraph(df_rt, 'results/speedup.png')

# ── incremental vs non-incremental plots ─────────────────────────────────

INC_COLORS = {
    'incremental':     '#2ca02c',
    'non-incremental': '#d62728',
}


def _inc_speedup(df_inc, filename):
    """Bar chart: speedup of incremental over non-incremental per variant/size."""
    variants = df_inc['Variant'].unique()
    graph_types = [gt for gt in ['LFR', 'ScaleFree'] if gt in df_inc['GraphType'].values]

    fig, axes = plt.subplots(1, len(graph_types),
                             figsize=(7 * len(graph_types), 5),
                             sharey=True, squeeze=False)
    variant_colors = {v: IMPL_COLORS.get(v, f'C{i}') for i, v in enumerate(variants)}

    for idx, gt in enumerate(graph_types):
        ax = axes[0][idx]
        dt = df_inc[df_inc['GraphType'] == gt]

        for j, variant in enumerate(variants):
            dv = dt[dt['Variant'] == variant]
            inc = dv[dv['Mode'] == 'incremental'][['Nodes', 'Time']].rename(
                columns={'Time': 't_inc'})
            noinc = dv[dv['Mode'] == 'non-incremental'][['Nodes', 'Time']].rename(
                columns={'Time': 't_noinc'})
            merged = pd.merge(inc, noinc, on='Nodes').dropna()
            if merged.empty:
                continue
            speedup = merged['t_noinc'] / merged['t_inc']
            ax.plot(merged['Nodes'], speedup,
                    marker='|', markersize=6, linewidth=1,
                    color=variant_colors.get(variant, '#333'), label=variant)

        ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_title(f'{gt} Graphs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    axes[0][0].set_ylabel('Speedup (non-incremental / incremental)', fontweight='bold')
    fig.suptitle('Incremental Quality Function Speedup', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


def _inc_runtime(df_inc, filename):
    """Grouped bar chart: runtime of inc vs non-inc per variant and graph size."""
    graph_types = [gt for gt in ['LFR', 'ScaleFree'] if gt in df_inc['GraphType'].values]
    variants = list(df_inc['Variant'].unique())

    fig, axes = plt.subplots(len(graph_types), 1,
                             figsize=(max(8, len(variants) * 3), 5 * len(graph_types)),
                             squeeze=False)

    for idx, gt in enumerate(graph_types):
        ax = axes[idx][0]
        dt = df_inc[df_inc['GraphType'] == gt]
        sizes = sorted(dt['Nodes'].unique())

        # Build grouped bars: x = variant×size, grouped by inc/noinc
        x_labels = []
        inc_times, noinc_times = [], []
        inc_errs, noinc_errs = [], []

        for n in sizes:
            for variant in variants:
                row_inc = dt[(dt['Variant'] == variant) & (dt['Mode'] == 'incremental') & (dt['Nodes'] == n)]
                row_noinc = dt[(dt['Variant'] == variant) & (dt['Mode'] == 'non-incremental') & (dt['Nodes'] == n)]
                t_i = row_inc['Time'].values[0] if len(row_inc) > 0 else float('nan')
                t_n = row_noinc['Time'].values[0] if len(row_noinc) > 0 else float('nan')
                te_i = row_inc['Time_Std'].values[0] if len(row_inc) > 0 else 0
                te_n = row_noinc['Time_Std'].values[0] if len(row_noinc) > 0 else 0
                inc_times.append(t_i)
                noinc_times.append(t_n)
                inc_errs.append(te_i)
                noinc_errs.append(te_n)
                short = variant.replace('BGL ', '')
                x_labels.append(f"{short}\nn={n:,}")

        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w/2, inc_times, w, yerr=inc_errs, label='incremental',
               color=INC_COLORS['incremental'], alpha=0.8, capsize=3)
        ax.bar(x + w/2, noinc_times, w, yerr=noinc_errs, label='non-incremental',
               color=INC_COLORS['non-incremental'], alpha=0.8, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_title(f'{gt} Graphs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Incremental vs Non-Incremental Runtime', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


def _inc_correctness(df_inc, filename):
    """Grouped bar chart: modularity of inc vs non-inc per variant and graph type."""
    graph_types = [gt for gt in ['LFR', 'ScaleFree'] if gt in df_inc['GraphType'].values]
    variants = list(df_inc['Variant'].unique())

    fig, axes = plt.subplots(len(graph_types), 1,
                             figsize=(max(8, len(variants) * 3), 5 * len(graph_types)),
                             squeeze=False)

    for idx, gt in enumerate(graph_types):
        ax = axes[idx][0]
        dt = df_inc[df_inc['GraphType'] == gt]
        sizes = sorted(dt['Nodes'].unique())

        x_labels = []
        inc_q, noinc_q = [], []
        inc_qe, noinc_qe = [], []

        for n in sizes:
            for variant in variants:
                row_inc = dt[(dt['Variant'] == variant) & (dt['Mode'] == 'incremental') & (dt['Nodes'] == n)]
                row_noinc = dt[(dt['Variant'] == variant) & (dt['Mode'] == 'non-incremental') & (dt['Nodes'] == n)]
                q_i = row_inc['Modularity'].values[0] if len(row_inc) > 0 else float('nan')
                q_n = row_noinc['Modularity'].values[0] if len(row_noinc) > 0 else float('nan')
                qe_i = row_inc['Modularity_Std'].values[0] if len(row_inc) > 0 else 0
                qe_n = row_noinc['Modularity_Std'].values[0] if len(row_noinc) > 0 else 0
                inc_q.append(q_i)
                noinc_q.append(q_n)
                inc_qe.append(qe_i)
                noinc_qe.append(qe_n)
                short = variant.replace('BGL ', '')
                x_labels.append(f"{short}\nn={n:,}")

        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w/2, inc_q, w, yerr=inc_qe, label='incremental',
               color=INC_COLORS['incremental'], alpha=0.8, capsize=3)
        ax.bar(x + w/2, noinc_q, w, yerr=noinc_qe, label='non-incremental',
               color=INC_COLORS['non-incremental'], alpha=0.8, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Modularity (Q)', fontweight='bold')
        ax.set_title(f'{gt} Graphs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Incremental vs Non-Incremental Modularity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, filename)


# Incremental vs Non-Incremental
if os.path.exists('results/incremental.csv'):
    df_inc = pd.read_csv('results/incremental.csv')
    print(f"Loaded {len(df_inc)} rows from results/incremental.csv")
    print(f"Variants: {', '.join(df_inc['Variant'].unique())}")
    print(f"Modes: {', '.join(df_inc['Mode'].unique())}")
    print()

    _inc_speedup(df_inc, 'results/inc_speedup.png')
    _inc_runtime(df_inc, 'results/inc_runtime.png')
    _inc_correctness(df_inc, 'results/inc_correctness.png')
