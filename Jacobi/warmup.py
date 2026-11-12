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
from torch.nn import functional as F


class MyMachine(nn.Module):

  def __init__(self, nt=10, datatype=torch.float16):
    super(MyMachine, self).__init__()
    self.datatype = datatype
    self.nt = int(nt)

    kernel = torch.tensor(
        [[0.0, 0.25, 0.0],
         [0.25, 0.0, 0.25],
         [0.0, 0.25, 0.0]],
        dtype=self.datatype,
    ).view(1, 1, 3, 3)
    self.register_buffer("kernel", kernel)

  def forward(self, X, X_prev, Mask):
    x = X
    mask = Mask

    for _ in range(self.nt):
      x_prev = x.clone()
      x_next = F.conv2d(x_prev, self.kernel, padding=1)
      x = x_next * mask

    return x


def main(warmup_runs=50):
  print("[INFO] Warming up NVIDIA GPU for Jacobi...")

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; an NVIDIA GPU is required")

  device = torch.device("cuda")
  capability = torch.cuda.get_device_capability(device)
  gpu_name = torch.cuda.get_device_name(device)
  print(f"[INFO] GPU: {gpu_name}  (SM {capability[0]}.{capability[1]})")

  # Enable TF32 if available (Ampere+)
  if capability >= (8, 0):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

  model = MyMachine(nt=10, datatype=torch.float16).to(device).eval()

  print("[INFO] Creating warmup Jacobi inputs (1024x1024, fp16)...")
  grid_size = 1024
  coordinates = torch.linspace(0, 1, grid_size, dtype=torch.float16, device=device)
  x_coords, y_coords = torch.meshgrid(coordinates, coordinates, indexing="ij")

  initial = torch.exp(
      -50.0 * ((x_coords - 0.5) ** 2 + (y_coords - 0.5) ** 2)
  ).unsqueeze(0).unsqueeze(0)
  x_prev = initial.clone()

  mask = torch.ones_like(initial)
  mask[:, :, 0, :] = 0
  mask[:, :, -1, :] = 0
  mask[:, :, :, 0] = 0
  mask[:, :, :, -1] = 0

  print(f"[INFO] Running {warmup_runs} warmup iterations...")
  with torch.inference_mode():
    for _ in range(warmup_runs):
      _ = model(initial, x_prev, mask)
  torch.cuda.synchronize(device)

  print("+++ Warmup Done +++")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Warm up the NVIDIA GPU before running Jacobi benchmarks."
  )
  parser.add_argument(
      "--runs",
      type=int,
      default=50,
      help="number of warmup iterations (default: 50)",
  )

  args = parser.parse_args()
  main(warmup_runs=args.runs)
