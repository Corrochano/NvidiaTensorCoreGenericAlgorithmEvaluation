"""
Copyright 2026 Álvaro Corrochano López

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import torch
from torch import nn


class MyMachine(nn.Module):
    def __init__(self):
        super(MyMachine, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)


def main(warmup_runs=100):
    print("[INFO] Warming up NVIDIA GPU...")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; an NVIDIA GPU is required")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[INFO] GPU: {gpu_name}  (SM {capability[0]}.{capability[1]})")

    # Enable TF32 if available (Ampere+), harmless on older GPUs
    if capability >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = MyMachine().to(device).eval()

    # Warmup with FP16 1024×1024 (same as the original OpenVINO warmup)
    print("[INFO] Creating warmup GEMM inputs (1024×1024, fp16)...")
    matrix_a = torch.rand(1024, 1024, dtype=torch.float16, device=device)
    matrix_b = torch.rand(1024, 1024, dtype=torch.float16, device=device)

    print(f"[INFO] Running {warmup_runs} warmup iterations...")
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(matrix_a, matrix_b)
    torch.cuda.synchronize(device)

    print("+++ Warmup Done +++")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Warm up the NVIDIA GPU before benchmarking."
    )
    parser.add_argument(
        "--runs", type=int, default=100,
        help="number of warmup iterations (default: 100)",
    )

    args = parser.parse_args()
    main(warmup_runs=args.runs)
