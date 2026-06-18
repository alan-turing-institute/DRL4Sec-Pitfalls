#!/bin/bash
# setup-envs.sh
# Creates one Python venv per case study in reproducibility/.venvs/:
#   .venvs/autorobust   (Python 3.10)
#   .venvs/link         (Python 3.9)
#   .venvs/sqirl        (Python 3.9)
#   .venvs/minicage     (Python 3.10)
#
# The four case studies cannot share an env (Python version, numpy 1.x vs 2.x,
# stable-baselines v1 vs SB3, etc.), so each gets its own venv.
#
# Prerequisite: python3.9 and python3.10 binaries must be on PATH.
#   - macOS:  brew install python@3.9 python@3.10
#             (or use pyenv: pyenv install 3.9.20 3.10.15)
#   - Linux:  use deadsnakes PPA, pyenv, or your distro's packages
#
# Usage: ./setup-envs.sh [autorobust|link|sqirl|minicage|all]
#   default = all

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVS_DIR="$SCRIPT_DIR/envs"
VENVS_DIR="$SCRIPT_DIR/.venvs"

mkdir -p "$VENVS_DIR"

# Locate a python binary whose reported version matches the requested major.minor.
# Tries the obvious aliases first, then falls back to pyenv shims.
find_python() {
    local want=$1   # e.g. "3.10"
    for cmd in "python$want" "python${want//./}" python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local got
            got=$("$cmd" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
            if [ "$got" = "$want" ]; then
                command -v "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

require_python() {
    local want=$1
    local found
    if ! found=$(find_python "$want"); then
        echo "ERROR: python$want not found on PATH." >&2
        echo "  Install it (e.g. \`brew install python@$want\` on macOS, or use pyenv)" >&2
        echo "  and re-run this script." >&2
        exit 1
    fi
    echo "$found"
}

make_venv() {
    local name=$1        # e.g. "autorobust"
    local pyver=$2       # e.g. "3.10"
    local reqfile=$3     # e.g. "$ENVS_DIR/autorobust.txt"
    local venv_path="$VENVS_DIR/$name"

    echo "============================================"
    echo " Creating venv: .venvs/$name (Python $pyver)"
    echo "============================================"

    local pybin
    pybin=$(require_python "$pyver")
    echo "[$name] using $pybin"

    if [ -d "$venv_path" ]; then
        echo "[$name] venv already exists: refreshing pip deps only."
    else
        "$pybin" -m venv "$venv_path"
    fi

    # shellcheck disable=SC1091
    "$venv_path/bin/pip" install --upgrade pip
    "$venv_path/bin/pip" install -r "$reqfile"
}

setup_autorobust() { make_venv autorobust 3.10 "$ENVS_DIR/autorobust.txt"; }
setup_link()       { make_venv link       3.9  "$ENVS_DIR/link.txt"; }
setup_sqirl()      { make_venv sqirl      3.9  "$ENVS_DIR/sqirl.txt"; }

setup_minicage() {
    make_venv minicage 3.10 "$ENVS_DIR/minicage.txt"
    # rl-reliability-metrics is an out-of-tree dep that must be cloned + installed.
    local eval_dir="$REPO_ROOT/MiniCAGE-Case-Study/Case-Study-Evaluation"
    if [ ! -d "$eval_dir/rl-reliability-metrics" ]; then
        echo "[minicage] Cloning rl-reliability-metrics..."
        git clone https://github.com/google-research/rl-reliability-metrics.git "$eval_dir/rl-reliability-metrics"
    fi
    "$VENVS_DIR/minicage/bin/pip" install -e "$eval_dir/rl-reliability-metrics/"
}

TARGETS="${1:-all}"
case "$TARGETS" in
    autorobust) setup_autorobust ;;
    link)       setup_link ;;
    sqirl)      setup_sqirl ;;
    minicage)   setup_minicage ;;
    all)
        setup_autorobust
        setup_link
        setup_sqirl
        setup_minicage
        ;;
    *)
        echo "Unknown target: $TARGETS" >&2
        echo "Usage: $0 [autorobust|link|sqirl|minicage|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo " Environment setup complete."
echo "============================================"
echo ""
echo "Venvs live in: $VENVS_DIR"
echo "Run reproduce-one-seed.sh or reproduce-full.sh next."
