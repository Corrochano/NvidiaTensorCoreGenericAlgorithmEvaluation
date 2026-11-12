#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

runs="${RUNS:-1}"
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer.\n' >&2
    exit 2
fi

# Verify CUDA is available before starting
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    printf '[ERROR] CUDA is not available. An NVIDIA GPU is required.\n' >&2
    exit 1
fi

gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
printf '[INFO] GPU detected: %s\n' "$gpu_name"

# --- Warmup -----------------------------------------------------------------
printf '\n[INFO] Running GPU warmup...\n'
python3 "$script_dir/warmup.py"

# --- Matrix sizes, datatypes, and execution modes ---------------------------
sizes=(256 512 1024 2048 4096 8192)
datatypes=(fp32 fp16 bf16)
exec_modes=(cpu gpu tensorcores)
failures=0

for size in "${sizes[@]}"; do
    for datatype in "${datatypes[@]}"; do
        for mode in "${exec_modes[@]}"; do
            printf '\n[INFO] Running %s GEMM %dx%d %s (%s runs)\n' \
                "$mode" "$size" "$size" "$datatype" "$runs"
            if ! python3 "$script_dir/executeInNvidia.py" \
                "$size" 1 "$runs" "$datatype" "$mode"; then
                printf '[ERROR] Failed: %s %dx%d %s\n' \
                    "$mode" "$size" "$size" "$datatype" >&2
                failures=$((failures + 1))
            fi
        done
    done
done

if (( failures > 0 )); then
    printf '\n[WARN] %d GEMM run(s) failed (some dtype/mode combos may not be supported on this GPU).\n' "$failures" >&2
fi

printf '\n+++ All GEMM runs completed +++\n'
