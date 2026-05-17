#!/usr/bin/env bash
# Build a HIP shared lib + post-process the assembly for kernel inspection.
# Prefers _fast.cpp (local scratch produced by iter_buffer_load.sh) and
# falls back to _b.cpp (tracked) so the script works on a fresh checkout.
set -euo pipefail

SRC="_fast.cpp"
if [[ ! -f "$SRC" ]]; then
    SRC="_b.cpp"
fi
if [[ ! -f "$SRC" ]]; then
    echo "build.sh: neither _fast.cpp (local scratch) nor _b.cpp (tracked) exists" >&2
    exit 1
fi

STEM="${SRC%.cpp}"
echo "build.sh: compiling $SRC"
hipcc -std=c++17 -fPIC --offload-arch=gfx950 --shared "$SRC" \
    -Rpass-analysis=kernel-resource-usage \
    -I/root/tilelang2/3rdparty/composable_kernel/include \
    --save-temps -g \
    -I/root/tilelang2/3rdparty/../src \
    -o _b.so

ASM_GLOB="${STEM}*-gfx950.s"
shopt -s nullglob
ASM_FILES=( $ASM_GLOB )
shopt -u nullglob
if (( ${#ASM_FILES[@]} > 0 )); then
    python gen_pure.py "${ASM_FILES[@]}"
else
    echo "build.sh: no $ASM_GLOB to post-process (gen_pure.py skipped)" >&2
fi