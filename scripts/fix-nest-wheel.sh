#!/usr/bin/env bash
#
# fix-nest-wheel.sh — re-apply workarounds for the broken nest-simulator PyPI wheel.
#
# The PyPI nest-simulator wheel (3.10.0rc*) ships `nest-config` and
# `build_info["prefix"]` with the wheel-build's temp paths frozen in, so it
# cannot be used to compile external NESTML modules out of the box.
# See: https://github.com/nest/nest-simulator/issues/3762
#
# `uv` rebuilds .venv/ on every sync, wiping manual fixes — so run this after
# `uv sync` (the Makefile's `sync` target does it for you).
#
# Idempotent and path-agnostic: safe to run repeatedly, and becomes a near
# no-op once NEST ships a fixed wheel. Works on macOS and Linux; the macOS-only
# steps (Homebrew libomp paths, dylib repointing) are skipped automatically
# elsewhere.
#
# Usage:
#   scripts/fix-nest-wheel.sh            # uses ./.venv (the workshop venv)
#   scripts/fix-nest-wheel.sh /path/venv # or a specific venv dir
#   PYTHON=/path/to/bin/python scripts/fix-nest-wheel.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."   # workshop project root

OS="$(uname -s)"

# --- locate the Python interpreter and NEST -------------------------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -n "${1:-}" ] && [ -x "$1/bin/python" ]; then
        PY="$1/bin/python"                 # explicit venv dir argument
    elif [ -x ".venv/bin/python" ]; then
        PY="$PWD/.venv/bin/python"         # the workshop venv (Makefile default)
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PY="$VIRTUAL_ENV/bin/python"
    else
        PY="$(command -v python3 || command -v python || true)"
    fi
fi
if [ -z "$PY" ] || ! "$PY" -c 'import nest' >/dev/null 2>&1; then
    echo "fix-nest-wheel: no python with 'nest' importable found — run 'uv sync' first" >&2
    exit 1
fi

# `import nest` prints a banner to stdout, so keep only the last line.
NEST_DIR="$("$PY" -c 'import nest, os; print(os.path.dirname(nest.__file__))' 2>/dev/null | tail -1)"
BINDIR="$(dirname "$PY")"
NEST_CONFIG="$BINDIR/nest-config"
[ ! -f "$NEST_CONFIG" ] && NEST_CONFIG="$(command -v nest-config 2>/dev/null || true)"
echo "fix-nest-wheel: nest package at $NEST_DIR"

if [ -z "$NEST_CONFIG" ] || [ ! -f "$NEST_CONFIG" ]; then
    echo "fix-nest-wheel: nest-config not found next to $PY — cannot patch" >&2
    exit 1
fi

# Portable in-place edit (works with both GNU and BSD sed).
replace_in_file() {  # <file> <sed-expr>
    local f="$1" expr="$2" tmp
    tmp="$(mktemp)"
    sed "$expr" "$f" > "$tmp" && mv "$tmp" "$f"
    chmod +x "$f"
}

# --- 1. nest-config: frozen build-temp prefix -> real install prefix --------
CURRENT_PREFIX="$(sh "$NEST_CONFIG" --prefix 2>/dev/null || true)"
if [ -n "$CURRENT_PREFIX" ] && [ ! -d "$CURRENT_PREFIX" ]; then
    echo "  [1] prefix: $CURRENT_PREFIX -> $NEST_DIR"
    replace_in_file "$NEST_CONFIG" "s|$CURRENT_PREFIX|$NEST_DIR|g"
else
    echo "  [1] prefix: ok"
fi

# --- 2. (macOS) nest-config: correct the Homebrew libomp paths --------------
if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1 && brew --prefix libomp >/dev/null 2>&1; then
    LIBOMP="$(brew --prefix libomp)"
    if ! sh "$NEST_CONFIG" --includes | grep -q -- "$LIBOMP/include"; then
        echo "  [2] includes: + -I$LIBOMP/include"
        replace_in_file "$NEST_CONFIG" "s| -I\([^ ]*\)/include/nest| -I$LIBOMP/include -I\1/include/nest|"
    else
        echo "  [2] includes: ok"
    fi
    if ! sh "$NEST_CONFIG" --libs | grep -q -- "$LIBOMP/lib/libomp.dylib"; then
        echo "  [2] libs: libomp.dylib -> $LIBOMP/lib/libomp.dylib"
        replace_in_file "$NEST_CONFIG" "s|[^ ]*/libomp\.dylib|$LIBOMP/lib/libomp.dylib|g"
    else
        echo "  [2] libs: ok"
    fi
else
    echo "  [2] libomp: not macOS/Homebrew, skipping"
fi

# --- 3. nest-config: symlink under <prefix>/bin (NESTML auto-detect) --------
mkdir -p "$NEST_DIR/bin"
ln -sf "$(cd "$(dirname "$NEST_CONFIG")" && pwd)/$(basename "$NEST_CONFIG")" "$NEST_DIR/bin/nest-config"
echo "  [3] symlink: $NEST_DIR/bin/nest-config"

# --- 4. compiled dynaple module: repoint OpenMP/ltdl, put on search path ----
MODULE="../dynaple-model-nest/neuron_model/target/dynaple_module.so"
if [ "$OS" = "Darwin" ] && [ -f "$MODULE" ]; then
    BUNDLED_OMP="$NEST_DIR/.dylibs/libomp.dylib"
    BUNDLED_LTDL="$NEST_DIR/.dylibs/libltdl.7.dylib"
    for dep in $(otool -L "$MODULE" | awk 'NR>1 {print $1}'); do
        case "$dep" in
            */libomp.dylib)
                if [ "$dep" != "$BUNDLED_OMP" ] && [ -f "$BUNDLED_OMP" ]; then
                    install_name_tool -change "$dep" "$BUNDLED_OMP" "$MODULE"
                fi
                ;;
            */libltdl.*.dylib)
                if [ "$dep" != "$BUNDLED_LTDL" ] && [ -f "$BUNDLED_LTDL" ]; then
                    install_name_tool -change "$dep" "$BUNDLED_LTDL" "$MODULE"
                fi
                ;;
        esac
    done
    mkdir -p "$NEST_DIR/lib/nest"
    cp "$MODULE" "$NEST_DIR/lib/nest/"
    echo "  [4] module: repointed + copied to $NEST_DIR/lib/nest/"
elif [ -f "$MODULE" ]; then
    # Linux: no libomp duplication issue; just make the module discoverable.
    mkdir -p "$NEST_DIR/lib/nest"
    cp "$MODULE" "$NEST_DIR/lib/nest/"
    echo "  [4] module: copied to $NEST_DIR/lib/nest/"
else
    echo "  [4] module: not built yet (run 'make build'), skipping"
fi

echo "fix-nest-wheel: done"
