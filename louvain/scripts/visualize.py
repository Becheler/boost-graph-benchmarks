#!/usr/bin/env python3
"""Visualize Louvain benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

if not os.path.exists('results/correctness.csv'):
    print("Error: results/correctness.csv not found")
    exit(1)

df = pd.read_csv('results/correctness.csv')

print(f"Loaded {len(df)} data points from results/correctness.csv")
print(f"Graphs: {', '.join(df['Graph'].unique())}")
print(f"Implementations: {', '.join(df['Implementation'].unique())}")
print()

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(16, 18))
gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.35, height_ratios=[1.5, 1.2, 1.2, 1.5, 1.5, 1.0])

# Modularity per graph (horizontal boxplots)
ax1a = fig.add_subplot(gs[0, 0])

implementations = ['Networkx', 'Igraph', 'Genlouvain', 'BGL']
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

graphs = df['Graph'].unique()
positions = []
data_to_plot = []
labels = []
box_colors = []

for i, graph in enumerate(graphs):
    for j, impl in enumerate(implementations):
        subset = df[(df['Graph'] == graph) & (df['Implementation'] == impl)]
        if len(subset) > 0:
            data_to_plot.append(subset['Modularity'].values)
            positions.append(i * (len(implementations) + 1) + j)
            labels.append(impl if i == 0 else '')
            box_colors.append(colors[j])

bp = ax1a.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                  showfliers=False, medianprops=dict(linewidth=1.5), vert=False)

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_alpha(0.7)

for i, color in enumerate(box_colors):
    bp['whiskers'][i*2].set_color(color)
    bp['whiskers'][i*2+1].set_color(color)
    bp['caps'][i*2].set_color(color)
    bp['caps'][i*2+1].set_color(color)
    bp['medians'][i].set_color(color)
    bp['medians'][i].set_linewidth(2)

graph_positions = [(i * (len(implementations) + 1) + 1.5) for i in range(len(graphs))]
ax1a.set_yticks(graph_positions)
ax1a.set_yticklabels(graphs, fontsize=9)
ax1a.set_xlabel('Modularity (Q)', fontweight='bold')
ax1a.set_title('Modularity per Graph (across implementations)', fontsize=12, fontweight='bold')
ax1a.grid(axis='x', alpha=0.3)
ax1a.invert_yaxis()

legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=impl) 
                   for i, impl in enumerate(implementations)]
ax1a.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=1)

# Communities per graph (horizontal boxplots)
ax1b = fig.add_subplot(gs[0, 1])

positions_comm = []
data_comm_to_plot = []
box_colors_comm = []

for i, graph in enumerate(graphs):
    for j, impl in enumerate(implementations):
        subset = df[(df['Graph'] == graph) & (df['Implementation'] == impl)]
        if len(subset) > 0 and 'Communities' in subset.columns:
            comm_values = subset['Communities'].dropna().values
            if len(comm_values) > 0:
                data_comm_to_plot.append(comm_values)
                positions_comm.append(i * (len(implementations) + 1) + j)
                box_colors_comm.append(colors[j])

bp2 = ax1b.boxplot(data_comm_to_plot, positions=positions_comm, widths=0.6, patch_artist=True,
                   showfliers=False, medianprops=dict(linewidth=1.5), vert=False)

for patch, color in zip(bp2['boxes'], box_colors_comm):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_alpha(0.7)

for i, color in enumerate(box_colors_comm):
    bp2['whiskers'][i*2].set_color(color)
    bp2['whiskers'][i*2+1].set_color(color)
    bp2['caps'][i*2].set_color(color)
    bp2['caps'][i*2+1].set_color(color)
    bp2['medians'][i].set_color(color)
    bp2['medians'][i].set_linewidth(2)

ax1b.set_yticks(graph_positions)
ax1b.set_yticklabels(graphs, fontsize=9)
ax1b.set_xlabel('Number of Communities', fontweight='bold')
ax1b.set_title('Communities Found per Graph (across implementations)', fontsize=12, fontweight='bold')
ax1b.grid(axis='x', alpha=0.3)
ax1b.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1b.invert_yaxis()

legend_elements2 = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=impl) 
                    for i, impl in enumerate(implementations)]
ax1b.legend(handles=legend_elements2, loc='lower right', fontsize=8, ncol=1)

# Relative performance (horizontal bars)
ax2 = fig.add_subplot(gs[1, :])
df_mean = df.groupby(['Graph', 'Implementation'])['Modularity'].mean().reset_index()
df_grouped = df_mean.groupby('Graph')
relative_perf = []
for graph, group in df_grouped:
    mean_q = group['Modularity'].mean()
    for _, row in group.iterrows():
        rel = ((row['Modularity'] - mean_q) / mean_q) * 100
        relative_perf.append({'Graph': graph, 'Implementation': row['Implementation'], 'Relative': rel})
df_rel = pd.DataFrame(relative_perf)

pivot_rel = df_rel.pivot(index='Graph', columns='Implementation', values='Relative')
available_impls = [impl for impl in ['Networkx', 'Igraph', 'Genlouvain', 'BGL'] if impl in pivot_rel.columns]
pivot_rel = pivot_rel[available_impls]
impl_colors = [colors[i] for i, impl in enumerate(implementations) if impl in available_impls]

pivot_rel.plot(kind='barh', ax=ax2, color=impl_colors, width=0.75)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('((Implementation Q - Mean Q) / Mean Q) × 100', fontweight='bold')
ax2.set_title('Relative Modularity Performance', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Relative communities (horizontal bars)
ax3 = fig.add_subplot(gs[2, :])
df_comm_mean = df.groupby(['Graph', 'Implementation'])['Communities'].mean().reset_index()
df_comm_grouped = df_comm_mean.groupby('Graph')
relative_comm = []
for graph, group in df_comm_grouped:
    mean_c = group['Communities'].mean()
    for _, row in group.iterrows():
        rel = ((row['Communities'] - mean_c) / mean_c) * 100 if mean_c > 0 else 0
        relative_comm.append({'Graph': graph, 'Implementation': row['Implementation'], 'Relative': rel})
df_rel_comm = pd.DataFrame(relative_comm)

pivot_rel_comm = df_rel_comm.pivot(index='Graph', columns='Implementation', values='Relative')
available_impls_comm = [impl for impl in ['Networkx', 'Igraph', 'Genlouvain', 'BGL'] if impl in pivot_rel_comm.columns]
pivot_rel_comm = pivot_rel_comm[available_impls_comm]
impl_colors_comm = [colors[i] for i, impl in enumerate(implementations) if impl in available_impls_comm]

pivot_rel_comm.plot(kind='barh', ax=ax3, color=impl_colors_comm, width=0.75)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('((Implementation Communities - Mean Communities) / Mean Communities) × 100', fontweight='bold')
ax3.set_title('Relative Communities Performance', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# BGL vs reference implementations (horizontal bars)
ax4 = fig.add_subplot(gs[3, :])
comparisons = []
for graph in df_mean['Graph'].unique():
    bgl_q = df_mean[(df_mean['Graph'] == graph) & (df_mean['Implementation'] == 'BGL')]['Modularity'].values
    if len(bgl_q) == 0:
        continue
    bgl_q = bgl_q[0]
    
    for impl in ['Networkx', 'Igraph', 'Genlouvain']:
        ref_q = df_mean[(df_mean['Graph'] == graph) & (df_mean['Implementation'] == impl)]['Modularity'].values
        if len(ref_q) > 0:
            ref_q = ref_q[0]
            diff = ((bgl_q - ref_q) / ref_q) * 100
            comparisons.append({'Graph': graph, 'Implementation': f'+1% Benefit\nvs {impl}' if impl == 'Networkx' else f'vs {impl}', 'Difference': diff})

df_comp = pd.DataFrame(comparisons)
pivot_comp = df_comp.pivot(index='Graph', columns='Implementation', values='Difference')

column_order = ['+1% Benefit\nvs Networkx', 'vs Igraph', 'vs Genlouvain']
pivot_comp = pivot_comp[[col for col in column_order if col in pivot_comp.columns]]

pivot_comp.plot(kind='barh', ax=ax4, color=['#1f77b4', '#ff7f0e', '#d62728'], width=0.75)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('((BGL Q - Reference Q) / Reference Q) × 100', fontweight='bold')
ax4.set_title('BGL vs Reference Implementations', fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=9)
ax4.grid(axis='x', alpha=0.3)
ax4.invert_yaxis()

# Runtime plots
if os.path.exists('results/runtime.csv'):
    df_runtime = pd.read_csv('results/runtime.csv')
    
    for idx, graph_type in enumerate(['LFR', 'ScaleFree']):
        ax = fig.add_subplot(gs[4, idx])
        df_type = df_runtime[df_runtime['GraphType'] == graph_type]
        
        ax.errorbar(df_type['Nodes'], df_type['NetworkX_Time'],
                   yerr=df_type['NetworkX_Std'], marker='o',
                   label='NetworkX', capsize=5, linewidth=2, color='#1f77b4')
        ax.errorbar(df_type['Nodes'], df_type['Igraph_Time'],
                   yerr=df_type['Igraph_Std'], marker='s',
                   label='igraph', capsize=5, linewidth=2, color='#ff7f0e')
        ax.errorbar(df_type['Nodes'], df_type['BGL_Time'],
                   yerr=df_type['BGL_Std'], marker='^',
                   label='BGL', capsize=5, linewidth=2, color='#2ca02c')
        
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_title(f'{graph_type} Graphs: Runtime Scalability', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Community count plots
    for idx, graph_type in enumerate(['LFR', 'ScaleFree']):
        ax = fig.add_subplot(gs[5, idx])
        df_type = df_runtime[df_runtime['GraphType'] == graph_type]
        
        if 'NetworkX_Communities' in df_type.columns:
            ax.errorbar(df_type['Nodes'], df_type['NetworkX_Communities'],
                       yerr=df_type['NetworkX_Communities_Std'], marker='o',
                       label='NetworkX', capsize=5, linewidth=2, color='#1f77b4')
            ax.errorbar(df_type['Nodes'], df_type['Igraph_Communities'],
                       yerr=df_type['Igraph_Communities_Std'], marker='s',
                       label='igraph', capsize=5, linewidth=2, color='#ff7f0e')
            ax.errorbar(df_type['Nodes'], df_type['BGL_Communities'],
                       yerr=df_type['BGL_Communities_Std'], marker='^',
                       label='BGL', capsize=5, linewidth=2, color='#2ca02c')
            
            ax.set_xlabel('Number of Nodes', fontweight='bold')
            ax.set_ylabel('Communities Found', fontweight='bold')
            ax.set_title(f'{graph_type} Graphs: Communities Detected', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')

fig.suptitle('Louvain Community Detection: Implementation Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('results/benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("Saved results/benchmark_comparison.png")
plt.close()
