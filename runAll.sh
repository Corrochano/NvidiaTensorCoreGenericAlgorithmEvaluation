#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Default settings (can be overridden by environment variables, CLI flags, or positional arguments)
runs="${RUNS:-1}"
iterations="${ITERATIONS:-100}"
selected_benchmarks=()

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [RUNS] [ITERATIONS]

Executes all three benchmarks (GEMM, Jacobi, Multigrid) in sequence.

Arguments:
  RUNS                 Number of test runs per configuration (default: 1)
  ITERATIONS           Number of iterations for iterative solvers (default: 100)

Options:
  -r, --runs N         Set number of runs (overrides RUNS env var)
  -i, --iterations N   Set number of iterations (overrides ITERATIONS env var)
  -b, --benchmark NAME Run only specific benchmark: gemm, jacobi, multigrid, or all
                       (can be specified multiple times)
  -h, --help           Display this help message and exit

Environment Variables:
  RUNS                 Default number of runs (default: 1)
  ITERATIONS           Default number of iterations (default: 100)

Examples:
  ./$(basename "$0")                      # Run all 3 benchmarks with defaults
  ./$(basename "$0") 3 50                 # Run all 3 with RUNS=3, ITERATIONS=50
  RUNS=2 ITERATIONS=200 ./$(basename "$0") # Run using environment variables
  ./$(basename "$0") --runs 5 --benchmark gemm --benchmark jacobi
EOF
}

# Parse options and positional arguments
positional_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -r|--runs)
            if [[ $# -lt 2 ]]; then
                printf '[ERROR] %s requires an argument.\n' "$1" >&2
                exit 2
            fi
            runs="$2"
            shift 2
            ;;
        -i|--iterations)
            if [[ $# -lt 2 ]]; then
                printf '[ERROR] %s requires an argument.\n' "$1" >&2
                exit 2
            fi
            iterations="$2"
            shift 2
            ;;
        -b|--benchmark)
            if [[ $# -lt 2 ]]; then
                printf '[ERROR] %s requires an argument.\n' "$1" >&2
                exit 2
            fi
            selected_benchmarks+=("$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')")
            shift 2
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                positional_args+=("$1")
                shift
            done
            break
            ;;
        -*)
            printf '[ERROR] Unknown option: %s\n' "$1" >&2
            print_usage >&2
            exit 2
            ;;
        *)
            positional_args+=("$1")
            shift
            ;;
    esac
done

if [[ ${#positional_args[@]} -ge 1 ]]; then
    runs="${positional_args[0]}"
fi
if [[ ${#positional_args[@]} -ge 2 ]]; then
    iterations="${positional_args[1]}"
fi
if [[ ${#positional_args[@]} -gt 2 ]]; then
    printf '[ERROR] Too many positional arguments provided.\n' >&2
    print_usage >&2
    exit 2
fi

# Validation
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf '[ERROR] RUNS must be a positive integer (got "%s").\n' "$runs" >&2
    exit 2
fi

if [[ ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    printf '[ERROR] ITERATIONS must be a positive integer (got "%s").\n' "$iterations" >&2
    exit 2
fi

# Determine which benchmarks to run
run_gemm=false
run_jacobi=false
run_multigrid=false

if [[ ${#selected_benchmarks[@]} -eq 0 ]]; then
    run_gemm=true
    run_jacobi=true
    run_multigrid=true
else
    for bm in "${selected_benchmarks[@]}"; do
        case "$bm" in
            gemm)
                run_gemm=true
                ;;
            jacobi)
                run_jacobi=true
                ;;
            multigrid)
                run_multigrid=true
                ;;
            all)
                run_gemm=true
                run_jacobi=true
                run_multigrid=true
                ;;
            *)
                printf '[ERROR] Unknown benchmark: "%s". Valid choices: gemm, jacobi, multigrid, all\n' "$bm" >&2
                exit 2
                ;;
        esac
    done
fi

# Verify CUDA availability
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    printf '[ERROR] CUDA is not available. An NVIDIA GPU is required to run the benchmarks.\n' >&2
    exit 1
fi

gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)

# Export variables so child scripts inherit them
export RUNS="$runs"
export ITERATIONS="$iterations"

printf '======================================================================\n'
printf '           NVIDIA Tensor Core Generic Algorithm Benchmarks            \n'
printf '======================================================================\n'
printf '  GPU:        %s\n' "$gpu_name"
printf '  Runs:       %s\n' "$runs"
printf '  Iterations: %s (Jacobi & Multigrid)\n' "$iterations"
printf '  GEMM:       %s\n' "$([ "$run_gemm" = true ] && echo "YES" || echo "SKIP")"
printf '  Jacobi:     %s\n' "$([ "$run_jacobi" = true ] && echo "YES" || echo "SKIP")"
printf '  Multigrid:  %s\n' "$([ "$run_multigrid" = true ] && echo "YES" || echo "SKIP")"
printf '======================================================================\n\n'

format_duration() {
    local total_seconds=$1
    local mins=$(( total_seconds / 60 ))
    local secs=$(( total_seconds % 60 ))
    printf '%dm %02ds' "$mins" "$secs"
}

declare -A bm_status
declare -A bm_duration
overall_failures=0
total_start_time=$(date +%s)

run_suite() {
    local name="$1"
    local script_path="$2"

    printf '\n======================================================================\n'
    printf '>>> Starting Benchmark: %s\n' "$name"
    printf '======================================================================\n'

    local start_time
    start_time=$(date +%s)

    if bash "$script_path"; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        bm_status["$name"]="PASSED"
        bm_duration["$name"]="$elapsed"
        printf '\n>>> [%s] finished successfully in %s.\n' "$name" "$(format_duration "$elapsed")"
    else
        local exit_code=$?
        local end_time
        end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        bm_status["$name"]="FAILED (exit code $exit_code)"
        bm_duration["$name"]="$elapsed"
        overall_failures=$((overall_failures + 1))
        printf '\n>>> [%s] FAILED (exit code %d) after %s.\n' "$name" "$exit_code" "$(format_duration "$elapsed")" >&2
    fi
}

# Execute GEMM
if [[ "$run_gemm" == true ]]; then
    run_suite "GEMM" "$script_dir/GEMM/runGEMM.sh"
else
    bm_status["GEMM"]="SKIPPED"
    bm_duration["GEMM"]="0"
fi

# Execute Jacobi
if [[ "$run_jacobi" == true ]]; then
    run_suite "Jacobi" "$script_dir/Jacobi/runJacobi.sh"
else
    bm_status["Jacobi"]="SKIPPED"
    bm_duration["Jacobi"]="0"
fi

# Execute Multigrid
if [[ "$run_multigrid" == true ]]; then
    run_suite "Multigrid" "$script_dir/Multigrid/runMultigrid.sh"
else
    bm_status["Multigrid"]="SKIPPED"
    bm_duration["Multigrid"]="0"
fi

total_end_time=$(date +%s)
total_duration=$(( total_end_time - total_start_time ))

printf '\n======================================================================\n'
printf '                         BENCHMARK SUMMARY                            \n'
printf '======================================================================\n'
printf '%-15s %-25s %-12s\n' "Benchmark" "Status" "Duration"
printf '%-15s %-25s %-12s\n' "---------------" "-------------------------" "------------"

for name in "GEMM" "Jacobi" "Multigrid"; do
    printf '%-15s %-25s %-12s\n' \
        "$name" \
        "${bm_status[$name]}" \
        "$(format_duration "${bm_duration[$name]}")"
done

printf '%-15s %-25s %-12s\n' "---------------" "-------------------------" "------------"
printf 'Total elapsed time: %s\n' "$(format_duration "$total_duration")"

if (( overall_failures > 0 )); then
    printf '\n[WARN] Completed with %d benchmark suite failure(s).\n' "$overall_failures" >&2
    exit 1
else
    printf '\n[SUCCESS] All executed benchmarks finished successfully.\n'
    exit 0
fi
