#!/usr/bin/env python3
"""
Compare per-level passes and moves between BGL Louvain and gen-louvain.

Temporarily instruments both source files with debug stderr prints,
rebuilds, runs on a generated Barabási-Albert graph, parses the output,
reverts the patches, and displays a side-by-side comparison.

Usage:
    python3 scripts/compare_passes.py [--nodes N] [--seed S]

Defaults: --nodes 10000  --seed 42
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths (relative to louvain/) ─────────────────────────────────────────

# Allow override via env var (used in CI where the fork lives elsewhere)
_bgl_include = os.environ.get(
    'BGL_GRAPH_INCLUDE',
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'forks', 'graph', 'include'))
)
BGL_HEADER = os.path.join(_bgl_include, 'boost', 'graph', 'louvain_clustering.hpp')

GENLOUVAIN_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'vendor', 'gen-louvain',
    'gen-louvain', 'src', 'louvain.cpp'))

BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'build')


# ── patch helpers ────────────────────────────────────────────────────────

def _read(path):
    with open(path, 'r') as f:
        return f.read()


def _write(path, content):
    with open(path, 'w') as f:
        f.write(content)


# ---- BGL patch: insert a cerr line right after "pass_number++;" inside
#      the first (incremental) local_optimization_impl do-while body.
BGL_ANCHOR = '    } while (num_moves > 0 && (Q_new - Q) > min_improvement_inner);'
BGL_PATCH  = textwrap.dedent("""\
        std::cerr << "BGL_PASS: level=LEVEL"
                  << " pass=" << pass_number
                  << " moves=" << num_moves
                  << " dQ=" << (Q_new - Q) << "\\n";
    } while (num_moves > 0 && (Q_new - Q) > min_improvement_inner);""")

# We also need to inject a level counter.  The simplest approach:
# replace the first `do {` block preamble with one that has a static level counter.
BGL_LEVEL_ANCHOR = (
    '    weight_type Q_new = Q;\n'
    '    std::size_t num_moves = 0;\n'
    '    std::size_t pass_number = 0;'
)
BGL_LEVEL_PATCH = (
    '    weight_type Q_new = Q;\n'
    '    std::size_t num_moves = 0;\n'
    '    std::size_t pass_number = 0;\n'
    '    static int _bgl_level_counter = 0;\n'
    '    int _bgl_this_level = ++_bgl_level_counter;'
)


# ---- Genlouvain patch: insert cerr right before the while-condition
GL_ANCHOR = '  } while (nb_moves>0 && new_qual-cur_qual > eps_impr);'
GL_PATCH  = textwrap.dedent("""\
    std::cerr << "GL_PASS: level=LEVEL"
              << " pass=" << nb_pass_done
              << " moves=" << nb_moves
              << " dQ=" << (new_qual - cur_qual) << "\\n";
  } while (nb_moves>0 && new_qual-cur_qual > eps_impr);""")

# Genlouvain outer loop: inject level counter in main_louvain.cpp
GL_MAIN = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'vendor', 'gen-louvain',
    'gen-louvain', 'src', 'main_louvain.cpp'))


def patch_bgl(original):
    """Return patched content and whether it succeeded."""
    patched = original
    # Add level-counter variables
    if BGL_LEVEL_ANCHOR in patched:
        patched = patched.replace(BGL_LEVEL_ANCHOR, BGL_LEVEL_PATCH, 1)
    else:
        return None

    # Insert the cerr line (replace LEVEL with the runtime counter)
    tag = BGL_PATCH.replace('level=LEVEL', 'level=" << _bgl_this_level << "')
    if BGL_ANCHOR in patched:
        patched = patched.replace(BGL_ANCHOR, tag, 1)
    else:
        return None

    # Include <iostream> if not already present
    if '#include <iostream>' not in patched:
        patched = patched.replace('#ifndef BOOST_GRAPH_LOUVAIN',
                                  '#include <iostream>\n#ifndef BOOST_GRAPH_LOUVAIN', 1)

    return patched


def patch_genlouvain(original):
    """Patch louvain.cpp to emit per-pass info on stderr."""
    patched = original
    # We need level info — it lives in main_louvain.cpp (outer loop).
    # For simplicity we pass something from _this_ file: we use a static counter
    # that resets each time one_level() is called the first time after construction.
    # Instead, let's just tag level=0 here (the actual level is tracked from main).

    # Actually, the level info comes from the outer loop in main_louvain.cpp.
    # We'll patch _both_ louvain.cpp and main_louvain.cpp.
    # In louvain.cpp, just emit pass/moves/dQ with level=UNKNOWN.
    # In main_louvain.cpp, emit "GL_LEVEL: <n>" before each one_level() call.

    if GL_ANCHOR in patched:
        tag = GL_PATCH.replace('level=LEVEL', 'level=0')
        patched = patched.replace(GL_ANCHOR, tag, 1)
    else:
        return None

    if '#include <iostream>' not in patched:
        patched = '#include <iostream>\n' + patched

    return patched


GL_MAIN_LEVEL_ANCHOR = '    improvement = c.one_level();'
GL_MAIN_LEVEL_PATCH = (
    '    std::cerr << "GL_LEVEL: " << level << std::endl;\n'
    '    improvement = c.one_level();'
)


def patch_genlouvain_main(original):
    """Patch main_louvain.cpp to emit level markers."""
    if GL_MAIN_LEVEL_ANCHOR in original:
        return original.replace(GL_MAIN_LEVEL_ANCHOR, GL_MAIN_LEVEL_PATCH, 1)
    return None


# ── graph generation ─────────────────────────────────────────────────────

def generate_graph(n, seed):
    """Generate a Barabási-Albert graph and write to a temp edge-list file."""
    import networkx as nx
    G = nx.barabasi_albert_graph(n, 3, seed=seed)
    fd, path = tempfile.mkstemp(suffix='.txt', prefix='passes_graph_')
    with os.fdopen(fd, 'w') as f:
        f.write(f'{G.number_of_nodes()} {G.number_of_edges()}\n')
        for u, v in G.edges():
            f.write(f'{u} {v} 1.0\n')
    return path, G.number_of_nodes(), G.number_of_edges()


# ── run implementations ─────────────────────────────────────────────────

def run_bgl(graph_file, seed):
    """Run BGL vecS/vecS and capture stderr for pass info, stderr/stdout for Q and time."""
    exe = os.path.join(BUILD_DIR, 'bgl_louvain_vecS_vecS')
    result = subprocess.run(
        [exe, graph_file, str(seed)],
        capture_output=True, text=True, timeout=300,
    )
    return result


def run_genlouvain(graph_file, seed):
    """Convert graph and run gen-louvain, return subprocess result."""
    convert = os.path.join(BUILD_DIR, 'genlouvain_convert')
    louvain = os.path.join(BUILD_DIR, 'genlouvain_louvain')

    # Convert edge list to binary format
    gl_txt = graph_file.replace('.txt', '_gl.txt')
    with open(graph_file) as fin:
        lines = fin.readlines()
    with open(gl_txt, 'w') as fout:
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                fout.write(f'{parts[0]} {parts[1]}\n')

    gl_bin = graph_file.replace('.txt', '.bin')
    subprocess.run(
        [convert, '-i', gl_txt, '-o', gl_bin],
        capture_output=True, text=True, timeout=30, check=True,
    )

    result = subprocess.run(
        [louvain, gl_bin, '-l', '-1', '-q', '0'],
        capture_output=True, text=True, timeout=300,
    )

    # Cleanup temp files
    for f in [gl_txt, gl_bin]:
        if os.path.exists(f):
            os.remove(f)

    return result


# ── parsing ──────────────────────────────────────────────────────────────

def parse_bgl_output(result):
    """Parse BGL stderr for pass info, stdout for Q."""
    levels = {}  # level -> [(pass, moves, dQ), ...]
    Q = None
    elapsed = None

    for line in result.stderr.strip().split('\n'):
        m = re.match(r'BGL_PASS: level=(\d+) pass=(\d+) moves=(\d+) dQ=([\d.eE+\-]+)', line)
        if m:
            lv, ps, mv, dq = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            levels.setdefault(lv, []).append((ps, mv, dq))
        if line.startswith('LOUVAIN_TIME:'):
            elapsed = float(line.split(':')[1].strip())

    lines = result.stdout.strip().split('\n')
    if lines:
        try:
            Q = float(lines[0])
        except ValueError:
            pass

    return levels, Q, elapsed


def parse_genlouvain_output(result):
    """Parse genlouvain stderr for level markers and pass info."""
    levels = {}
    current_level = 0
    Q = None
    elapsed = None

    for line in result.stderr.strip().split('\n'):
        lm = re.match(r'GL_LEVEL:\s*(\d+)', line)
        if lm:
            current_level = int(lm.group(1)) + 1  # 1-based
            continue

        m = re.match(r'GL_PASS: level=\d+ pass=(\d+) moves=(\d+) dQ=([\d.eE+\-]+)', line)
        if m:
            ps, mv, dq = int(m.group(1)), int(m.group(2)), float(m.group(3))
            levels.setdefault(current_level, []).append((ps, mv, dq))

        if line.startswith('LOUVAIN_TIME:'):
            elapsed = float(line.split(':')[1].strip())

    # Q is last float on stderr (genlouvain prints modularity to stderr)
    for line in reversed(result.stderr.strip().split('\n')):
        line = line.strip()
        if line.startswith('LOUVAIN_TIME:') or line.startswith('GL_'):
            continue
        try:
            Q = float(line)
            break
        except ValueError:
            continue

    return levels, Q, elapsed


# ── display ──────────────────────────────────────────────────────────────

def print_comparison(bgl_levels, bgl_Q, bgl_time, gl_levels, gl_Q, gl_time, n, m):
    """Print a side-by-side comparison table."""
    print(f'\n{"=" * 72}')
    print(f'  Pass/Move Comparison  —  Barabási-Albert graph  (n={n}, m={m})')
    print(f'{"=" * 72}\n')

    all_levels = sorted(set(list(bgl_levels.keys()) + list(gl_levels.keys())))

    bgl_total_passes = 0
    bgl_total_moves = 0
    gl_total_passes = 0
    gl_total_moves = 0

    for lv in all_levels:
        bgl_p = bgl_levels.get(lv, [])
        gl_p = gl_levels.get(lv, [])

        bgl_passes = len(bgl_p)
        bgl_moves_lv = sum(mv for _, mv, _ in bgl_p)
        gl_passes = len(gl_p)
        gl_moves_lv = sum(mv for _, mv, _ in gl_p)

        bgl_total_passes += bgl_passes
        bgl_total_moves += bgl_moves_lv
        gl_total_passes += gl_passes
        gl_total_moves += gl_moves_lv

        print(f'  Level {lv}:')
        print(f'    {"":20s}  {"BGL":>10s}  {"genlouvain":>12s}  {"ratio":>8s}')
        print(f'    {"Passes":20s}  {bgl_passes:10d}  {gl_passes:12d}  '
              f'{bgl_passes / max(gl_passes, 1):8.1f}x')
        print(f'    {"Total moves":20s}  {bgl_moves_lv:10d}  {gl_moves_lv:12d}  '
              f'{bgl_moves_lv / max(gl_moves_lv, 1):8.1f}x')

        # First and last pass detail
        if bgl_p:
            print(f'    {"  pass 1 moves":20s}  {bgl_p[0][1]:10d}', end='')
        else:
            print(f'    {"  pass 1 moves":20s}  {"—":>10s}', end='')
        if gl_p:
            print(f'  {gl_p[0][1]:12d}')
        else:
            print(f'  {"—":>12s}')

        if bgl_p and len(bgl_p) > 1:
            print(f'    {"  last pass moves":20s}  {bgl_p[-1][1]:10d}', end='')
        elif bgl_p:
            print(f'    {"  last pass moves":20s}  {bgl_p[0][1]:10d}', end='')
        else:
            print(f'    {"  last pass moves":20s}  {"—":>10s}', end='')
        if gl_p and len(gl_p) > 1:
            print(f'  {gl_p[-1][1]:12d}')
        elif gl_p:
            print(f'  {gl_p[0][1]:12d}')
        else:
            print(f'  {"—":>12s}')
        print()

    print(f'  {"─" * 68}')
    print(f'  {"TOTAL":20s}  {"BGL":>10s}  {"genlouvain":>12s}  {"ratio":>8s}')
    print(f'  {"Passes":20s}  {bgl_total_passes:10d}  {gl_total_passes:12d}  '
          f'{bgl_total_passes / max(gl_total_passes, 1):8.1f}x')
    print(f'  {"Moves":20s}  {bgl_total_moves:10d}  {gl_total_moves:12d}  '
          f'{bgl_total_moves / max(gl_total_moves, 1):8.1f}x')
    print()
    print(f'  {"Modularity (Q)":20s}  {bgl_Q or 0:.6f}       {gl_Q or 0:.6f}')
    print(f'  {"Time (s)":20s}  {bgl_time or 0:.4f}         {gl_time or 0:.4f}')

    if bgl_time and gl_time:
        print(f'  {"Runtime ratio":20s}  {bgl_time / gl_time:.2f}x')
    if bgl_total_passes and gl_total_passes:
        print(f'  {"Pass ratio":20s}  {bgl_total_passes / gl_total_passes:.2f}x')

    print(f'\n{"=" * 72}\n')


# ── plotting ─────────────────────────────────────────────────────────────

def _flatten_with_levels(levels):
    """Flatten per-level pass data into (global_pass, dQ, level)."""
    seq = []
    gp = 0
    for lv in sorted(levels.keys()):
        for (ps, mv, dq) in levels[lv]:
            gp += 1
            seq.append((gp, max(dq, 0), lv))
    return seq


def _viridis_shades(low, high, n_levels):
    """Sample *n_levels* colours from Viridis between *low* and *high* (0-1)."""
    cmap = plt.cm.viridis
    positions = np.linspace(low, high, n_levels)
    return [cmap(p) for p in positions]


def plot_cumulative_dQ(bgl_levels, bgl_Q, gl_levels, gl_Q, n, m, outdir='results'):
    """Single-axis plot: remaining ΔQ curve + per-pass bars underneath.

    BGL uses the purple end of Viridis, gen-louvain the yellow/green end.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    all_level_ids = sorted(set(list(bgl_levels.keys()) + list(gl_levels.keys())))
    n_lvl = len(all_level_ids)

    # BGL → purple end (0.0–0.35), gen-louvain → yellow end (0.65–1.0)
    bgl_shades = _viridis_shades(0.0, 0.35, n_lvl)
    gl_shades  = _viridis_shades(0.65, 1.0, n_lvl)

    # Use the middle shade as the curve colour
    bgl_base = plt.cm.viridis(0.15)
    gl_base  = plt.cm.viridis(0.82)

    impl = [
        (bgl_levels, bgl_Q, 'BGL (eps = 0)',            bgl_base, -0.2, bgl_shades),
        (gl_levels,  gl_Q,  'gen-louvain (eps = 1e-6)', gl_base,   0.2, gl_shades),
    ]

    # ── remaining ΔQ area curves ─────────────────────────────────────
    for levels, Q_final, label, color, _, _ in impl:
        flat = _flatten_with_levels(levels)
        if not flat:
            continue
        passes = np.array([p[0] for p in flat])
        dqs    = np.array([p[1] for p in flat])
        cum    = np.cumsum(dqs)
        total  = cum[-1] if cum[-1] > 0 else 1.0
        remaining = (1.0 - cum / total) * 100

        ax.plot(passes, remaining, color=color, linewidth=1.2,
                label=f'{label}  ({len(flat)} passes, Q = {Q_final:.4f})')

    # ── per-pass ΔQ bars on twin y-axis ──────────────────────────────
    ax2 = ax.twinx()

    bar_width = 0.35
    legend_done = set()  # track "BGL L1", "GL L1", …

    for levels, Q_final, impl_label, base_color, offset, shades in impl:
        flat = _flatten_with_levels(levels)
        if not flat:
            continue
        total = sum(p[1] for p in flat) or 1.0
        short = impl_label.split('(')[0].strip()   # "BGL" or "gen-louvain"

        for gp, dq, lv in flat:
            idx = all_level_ids.index(lv)
            bar_color = shades[idx]
            key = f'{short} L{lv}'
            lbl = key if key not in legend_done else None
            ax2.bar(gp + offset, dq / total * 100, width=bar_width,
                    color=bar_color, alpha=0.7, edgecolor='white', linewidth=0.3,
                    label=lbl)
            if lbl:
                legend_done.add(key)

    ax2.set_ylabel('Per-pass ΔQ (% of total)', fontsize=8, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey', labelsize=7)
    ax2.set_ylim(bottom=0)

    # ── axes / labels ────────────────────────────────────────────────
    ax.set_xlabel('Global pass number (all levels)', fontsize=9)
    ax.set_ylabel('Remaining ΔQ (% of total gain)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0.5)
    ax.set_title(f'Diminishing returns of Louvain passes  —  '
                 f'Barabási-Albert  n = {n:,}   m = {m:,}',
                 fontsize=10)
    ax.grid(axis='both', alpha=0.12)

    # Merge legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right', ncol=2)

    fig.tight_layout()
    path = os.path.join(outdir, 'cumulative_dQ.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {path}')
    return path


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nodes', '-n', type=int, default=10000,
                        help='Number of nodes (default: 10000)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Ensure we're in the louvain/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    louvain_dir = os.path.dirname(script_dir)
    os.chdir(louvain_dir)

    # Verify source files exist
    for path, label in [(BGL_HEADER, 'BGL header'), (GENLOUVAIN_SRC, 'genlouvain louvain.cpp'),
                        (GL_MAIN, 'genlouvain main_louvain.cpp')]:
        if not os.path.exists(path):
            print(f'ERROR: {label} not found at {path}', file=sys.stderr)
            sys.exit(1)

    # Save originals
    bgl_orig = _read(BGL_HEADER)
    gl_orig = _read(GENLOUVAIN_SRC)
    gl_main_orig = _read(GL_MAIN)

    # Apply patches
    print('[1/5] Patching source files with debug prints …')
    bgl_patched = patch_bgl(bgl_orig)
    gl_patched = patch_genlouvain(gl_orig)
    gl_main_patched = patch_genlouvain_main(gl_main_orig)

    if not bgl_patched or not gl_patched or not gl_main_patched:
        print('ERROR: Could not locate patch anchors — source code may have changed.',
              file=sys.stderr)
        sys.exit(1)

    _write(BGL_HEADER, bgl_patched)
    _write(GENLOUVAIN_SRC, gl_patched)
    _write(GL_MAIN, gl_main_patched)

    try:
        # Rebuild
        print('[2/5] Rebuilding binaries …')
        rebuild = subprocess.run(
            ['make', '-j', '-C', 'build', 'bgl_louvain_vecS_vecS', 'genlouvain_louvain'],
            capture_output=True, text=True, timeout=120,
        )
        if rebuild.returncode != 0:
            print('Build failed:\n' + rebuild.stderr, file=sys.stderr)
            raise RuntimeError('build failed')

        # Generate graph
        print(f'[3/5] Generating Barabási-Albert graph (n={args.nodes}) …')
        graph_file, n_nodes, n_edges = generate_graph(args.nodes, args.seed)

        try:
            # Run BGL
            print('[4/5] Running BGL vecS/vecS …')
            bgl_result = run_bgl(graph_file, args.seed)
            bgl_levels, bgl_Q, bgl_time = parse_bgl_output(bgl_result)

            # Run genlouvain
            print('[5/5] Running gen-louvain …')
            gl_result = run_genlouvain(graph_file, args.seed)
            gl_levels, gl_Q, gl_time = parse_genlouvain_output(gl_result)

            # Display table
            print_comparison(bgl_levels, bgl_Q, bgl_time,
                             gl_levels, gl_Q, gl_time,
                             n_nodes, n_edges)

            # Plot cumulative ΔQ
            plot_cumulative_dQ(bgl_levels, bgl_Q, gl_levels, gl_Q,
                               n_nodes, n_edges)

        finally:
            if os.path.exists(graph_file):
                os.remove(graph_file)

    finally:
        # ALWAYS revert patches, even on failure
        print('Reverting source patches …')
        _write(BGL_HEADER, bgl_orig)
        _write(GENLOUVAIN_SRC, gl_orig)
        _write(GL_MAIN, gl_main_orig)

        # Rebuild clean binaries
        print('Rebuilding clean binaries …')
        subprocess.run(
            ['make', '-j', '-C', 'build', 'bgl_louvain_vecS_vecS', 'genlouvain_louvain'],
            capture_output=True, text=True, timeout=120,
        )
        # Reset BGL static counter for next run
        print('Done — sources reverted to original.')


if __name__ == '__main__':
    main()
