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
sm_major=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo 0)
printf '[INFO] GPU detected: %s (SM major: %s)\n' "$gpu_name" "$sm_major"

# --- Warmup -----------------------------------------------------------------
printf '\n[INFO] Running GPU warmup...\n'
python3 "$script_dir/warmup.py"

# --- Matrix sizes, datatypes, and execution modes ---------------------------
sizes=(256 512 1024 2048 4096 8192 12288 16384 20480 24576 28672 32768)
exec_modes=(${EXEC_MODES:-gpu tensorcores})
failures=0

for size in "${sizes[@]}"; do
    for mode in "${exec_modes[@]}"; do
        # 'gpu' mode: standard CUDA cores baseline (no Tensor Cores) -> FP32 (allow_tf32=False)
        # 'tensorcores' mode: Tensor Cores -> FP16 (SM 7+), plus TF32 (FP32) and BF16 if SM >= 8
        if [[ "$mode" == "gpu" ]]; then
            mode_datatypes=(fp32)
        elif [[ "$mode" == "tensorcores" ]]; then
            if (( sm_major >= 8 )); then
                mode_datatypes=(fp32 fp16 bf16)
            else
                mode_datatypes=(fp16)
            fi
        else
            mode_datatypes=(fp32)
        fi
        for datatype in "${mode_datatypes[@]}"; do
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
