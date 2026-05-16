#!/usr/bin/env bash
# Quick-iterate script for the ROCm buffer_load / swizzle-swap experiments.
#
# Usage:
#   scripts/iter_buffer_load.sh                # swap off (baseline)
#   scripts/iter_buffer_load.sh --swap         # swap on
#   scripts/iter_buffer_load.sh --swap -- shape args...   # custom shape
#
# What it does, in order:
#   1. cd into the tilelang checkout
#   2. wipe the tilelang JIT kernel cache (so source changes recompile)
#   3. rebuild tilelang editable (USE_ROCM=ON)
#   4. cd back to this repo
#   5. run the target gemm case via gemm/example_gemm.py-style invocation
#      (matmul_nt 8192x8192x16384 tile 256x256x64 stages=2 -- the target)
#
# Stops at the first failing step. Echo "[iter]" lines for grep.

set -euo pipefail
set -e

TILELANG_DIR="${TILELANG_DIR:-/root/tilelang2}"
BENCH_DIR="${BENCH_DIR:-/root/tile-kernel-bench-cdna4}"
EXECUTION_BACKEND="${TILELANG_EXECUTION_BACKEND:-cython}"

# Default shape/tile (the target case in the user's bench: 8192x8192x16384
# NT with block 256x256x64, num_stages=2).
M=8192
N=8192
K=2048
BLOCK_M=256
BLOCK_N=256
BLOCK_K=64
NUM_STAGES=2
NUM_THREADS=512

ENABLE_SWAP=1

# ---- arg parsing -----------------------------------------------------------
extra_python_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --swap)
      ENABLE_SWAP=1
      shift
      ;;
    --no-swap)
      ENABLE_SWAP=0
      shift
      ;;
    --shape)
      # --shape M,N,K
      IFS=',' read -r M N K <<<"$2"
      shift 2
      ;;
    --tile)
      # --tile bM,bN,bK
      IFS=',' read -r BLOCK_M BLOCK_N BLOCK_K <<<"$2"
      shift 2
      ;;
    --stages)
      NUM_STAGES="$2"
      shift 2
      ;;
    --threads)
      NUM_THREADS="$2"
      shift 2
      ;;
    --tilelang-dir)
      TILELANG_DIR="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-cache-wipe)
      SKIP_CACHE_WIPE=1
      shift
      ;;
    --)
      shift
      extra_python_args=("$@")
      break
      ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      echo "[iter] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done
sed -i 's/zipfile\.ZIP_DEFLATED/zipfile.ZIP_STORED/g' /opt/venv/lib/python3.10/site-packages/scikit_build_core/build/_wheelfile.py
# ---- step 1: wipe cache ----------------------------------------------------
if [[ -z "${SKIP_CACHE_WIPE:-}" ]]; then
  echo "[iter] wiping tilelang JIT cache"
  rm -rf /root/.tilelang/cache/0.1.9_*
else
  echo "[iter] (skip cache wipe)"
fi

# ---- step 2: rebuild tilelang ---------------------------------------------
if [[ -z "${SKIP_BUILD:-}" ]]; then
  echo "[iter] rebuilding tilelang in ${TILELANG_DIR}"
  pushd "${TILELANG_DIR}" >/dev/null
  USE_ROCM=ON VERBOSE=1 PYTHONUNBUFFERED=1 \
    pip install -e . --no-deps --no-build-isolation -v
  popd >/dev/null
else
  echo "[iter] (skip build)"
fi

# ---- step 3: run the target gemm ------------------------------------------
cd "${BENCH_DIR}"

if [[ "$ENABLE_SWAP" == "1" ]]; then
  export TL_ENABLE_ROCM_SWIZZLE_SWAP=1
  echo "[iter] TL_ENABLE_ROCM_SWIZZLE_SWAP=1 (swap ENABLED)"
else
  unset TL_ENABLE_ROCM_SWIZZLE_SWAP
  echo "[iter] swap DISABLED (baseline)"
fi

export TILELANG_EXECUTION_BACKEND="${EXECUTION_BACKEND}"
# Show hipcc command + its stdout/stderr when tilelang JIT-compiles a kernel.
export TILELANG_VERBOSE=1
export TILELANG_HIP_SAVE_TEMP_FILES=1
rm tmp*
echo "[iter] running NT ${M}x${N}x${K} tile ${BLOCK_M}x${BLOCK_N}x${BLOCK_K} stages=${NUM_STAGES} threads=${NUM_THREADS}"

python -c "
import sys, time
import tl_patches, torch
from gemm.example_gemm import matmul_nt

M, N, K = ${M}, ${N}, ${K}
bM, bN, bK = ${BLOCK_M}, ${BLOCK_N}, ${BLOCK_K}
ns = ${NUM_STAGES}
nt = ${NUM_THREADS}

t0 = time.time()
k = matmul_nt(M, N, K, bM, bN, bK, num_stages=ns, num_threads=nt)
print(f'[iter] compile: {time.time()-t0:.2f}s')

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(N, K, device='cuda', dtype=torch.float16)
c = k(a, b)
torch.cuda.synchronize()

# Correctness vs reference
try:
    ref = a @ b.T
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    correctness = 'PASS'
except AssertionError as e:
    correctness = 'FAIL: ' + str(e).split('\n')[0][:100]

# Bench
profiler = k.get_profiler()
latency_ms = profiler.do_bench(backend='cupti', input_tensors=[a, b])
flops = 2.0 * M * N * K
bytes_moved = (M*K + N*K + M*N) * 2
tflops = flops / (latency_ms * 1e-3) / 1e12
tbps = bytes_moved / (latency_ms * 1e-3) / 1e12

print(f'[iter] correctness: {correctness}')
print(f'[iter] latency: {latency_ms:.4f} ms')
print(f'[iter] TFLOPS:   {tflops:.2f}')
print(f'[iter] TB/s:     {tbps:.3f}')
print(f'[iter] VGPR:     {k.n_regs}')
print(f'[iter] spill+sc: {k.n_spills}')
"
python gen_pure.py tmp*-gfx950.s