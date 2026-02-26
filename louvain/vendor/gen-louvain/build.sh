#!/bin/bash
# Download and patch gen-louvain source for CMake integration.
# Source: https://sourceforge.net/projects/louvain/files/GenericLouvain/
# After running this script, the CMake build will compile gen-louvain automatically.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Skip if source already extracted and patched
if [ -d gen-louvain/src ] && [ -f gen-louvain/.patched ]; then
    echo "  gen-louvain source already prepared, skipping."
    echo "  (delete gen-louvain/ to force a re-download)"
    exit 0
fi

# --- Download ---
ARCHIVE="louvain-generic.tar.gz"
if [ ! -f "$ARCHIVE" ]; then
    echo "  Downloading gen-louvain from SourceForge..."
    curl -L -o "$ARCHIVE" \
        "https://sourceforge.net/projects/louvain/files/GenericLouvain/louvain-generic.tar.gz/download"
fi

# --- Extract ---
echo "  Extracting..."
rm -rf gen-louvain
tar xzf "$ARCHIVE"

# --- Fix Windows line endings (cross-platform sed -i) ---
if sed --version >/dev/null 2>&1; then
    # GNU sed (Linux)
    find gen-louvain -name '*.cpp' -o -name '*.h' | xargs sed -i 's/\r$//'
else
    # BSD sed (macOS)
    find gen-louvain -name '*.cpp' -o -name '*.h' | xargs sed -i '' 's/\r$//'
fi

# --- Patch: add std::chrono timing instrumentation ---
echo "  Applying timing patch..."
patch -p0 < timing.patch

# Mark as patched
touch gen-louvain/.patched
echo "  gen-louvain source prepared. CMake will compile it."
