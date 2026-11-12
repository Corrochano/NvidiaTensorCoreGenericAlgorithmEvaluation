#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

runs="${RUNS:-1}"
measure_energy="${MEASURE_ENERGY:-0}"
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer.\n' >&2
    exit 2
fi
if [[ "$measure_energy" != 0 && "$measure_energy" != 1 ]]; then
    printf 'MEASURE_ENERGY must be 0 or 1.\n' >&2
    exit 2
fi
if [[ "$measure_energy" == 1 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        printf 'sudo is required when MEASURE_ENERGY=1 (for RAPL CPU counters).\n' >&2
        exit 2
    fi
    sudo -v || exit 2
fi

# Verify CUDA is available before starting
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    printf '[ERROR] CUDA is not available. An NVIDIA GPU with Tensor Cores is required.\n' >&2
    exit 1
fi

gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
printf '[INFO] GPU detected: %s\n' "$gpu_name"

# --- Warmup -----------------------------------------------------------------
printf '\n[INFO] Running GPU warmup...\n'
python3 "$script_dir/warmup.py"

# --- Matrix sizes and datatypes ---------------------------------------------
sizes=(256 512 1024 2048 4096 8192 12288 16384 20480 24576 28672 32768)
datatypes=(fp32 fp16 bf16)
failures=0

for size in "${sizes[@]}"; do
    for datatype in "${datatypes[@]}"; do
        printf '\n[INFO] Running CUDA GEMM %dx%d %s (%s runs)\n' \
            "$size" "$size" "$datatype" "$runs"
        if [[ "$measure_energy" == 1 ]] && ! sudo -v; then
            printf '[ERROR] Could not refresh sudo authorization for RAPL reads.\n' >&2
            failures=$((failures + 1))
            continue
        fi
        if ! python3 "$script_dir/executeInNvidia.py" "$size" 1 "$runs" "$datatype"; then
            printf '[ERROR] Failed: CUDA %dx%d %s\n' "$size" "$size" "$datatype" >&2
            failures=$((failures + 1))
        fi
    done
done

if (( failures > 0 )); then
    printf '\n[ERROR] %d GEMM run(s) failed.\n' "$failures" >&2
    exit 1
fi

printf '\n+++ All GEMM runs completed +++\n'
