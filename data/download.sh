#!/usr/bin/env bash
# Download MNIST binary files into the current directory.
# Run from your build/data directory:
#   cd build/data && bash ../../data/download.sh

set -e
cd "$(dirname "$0")"

# PyTorch S3 mirror — reliable and fast
BASE="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for gz in "${FILES[@]}"; do
    out="${gz%.gz}"
    if [ -f "$out" ]; then
        echo "Already have $out — skipping."
        continue
    fi
    echo "Downloading $gz ..."
    curl -fsSL "$BASE/$gz" -o "$gz"
    echo "Extracting ..."
    gunzip "$gz"
    echo "  ✓ $out"
done

echo ""
echo "MNIST ready:"
ls -lh train-images-idx3-ubyte train-labels-idx1-ubyte \
        t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte
