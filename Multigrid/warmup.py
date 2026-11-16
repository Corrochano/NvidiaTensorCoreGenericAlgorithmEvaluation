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

  def __init__(self, nt=10, num_levels=10, datatype=torch.float16):
    super(MyMachine, self).__init__()
    self.datatype = datatype
    self.num_levels = int(num_levels)
    self.nt = int(nt)

    kernel = torch.tensor(
        [[0.0, 0.25, 0.0],
         [0.25, 0.0, 0.25],
         [0.0, 0.25, 0.0]],
        dtype=self.datatype,
    ).view(1, 1, 3, 3)
    self.register_buffer("kernel", kernel)

    restriction_kernel = torch.full(
        (1, 1, 2, 2),
        0.25,
        dtype=self.datatype,
    )
    self.register_buffer("restriction_kernel", restriction_kernel)

  def restriction(self, residual):
    return F.conv2d(residual, self.restriction_kernel, stride=2)

  def interpolate(self, f, target_size):
    return F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)

  def jacobi(self, z, mask):
    x = z
    for _ in range(self.nt):
      x_prev = x.clone()
      x_next = F.conv2d(x_prev, self.kernel, padding=1)
      x = x_next * mask
    return x

  def forward(
      self,
      X,
      Y,
      Mask1,
      Mask2,
      Mask3,
      Mask4,
      Mask5,
      Mask6,
      Mask7,
      Mask8,
      Mask9,
      Mask10,
  ):
    masks = [
        Mask1,
        Mask2,
        Mask3,
        Mask4,
        Mask5,
        Mask6,
        Mask7,
        Mask8,
        Mask9,
        Mask10,
    ]
    x = torch.exp(
        -50.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    ).to(self.datatype)
    x = x.unsqueeze(0).unsqueeze(0)

    grids = [x]
    num_levels = self.num_levels

    for level in range(1, num_levels):
      m = masks[level - 1]
      masked_input = grids[level - 1] * m
      unmasked_input = grids[level - 1] * (1.0 - m)

      masked_output = F.conv2d(masked_input, self.kernel, padding=1)
      unmasked_output = F.conv2d(unmasked_input, self.kernel, padding=1)

      residual = masked_output + unmasked_output
      coarse_residual = self.restriction(residual)
      grids.append(coarse_residual)

    coarse_solution = grids[-1]
    coarse_solution = self.jacobi(coarse_solution, masks[num_levels - 1])
    grids[-1] = coarse_solution

    for level in range(num_levels - 2, -1, -1):
      target_size = grids[level].shape[-2:]
      fine_solution = self.interpolate(grids[level + 1], target_size=target_size)
      fine_solution = fine_solution + grids[level]

      m = masks[level]
      masked_input = fine_solution * m
      unmasked_input = fine_solution * (1.0 - m)

      masked_output = F.conv2d(masked_input, self.kernel, padding=1)
      unmasked_output = F.conv2d(unmasked_input, self.kernel, padding=1)

      fine_solution = masked_output + unmasked_output
      grids[level] = fine_solution

    final_solution = grids[0]
    final_solution = self.jacobi(final_solution, Mask1)
    return final_solution


def main(warmup_runs=50):
  print("[INFO] Warming up NVIDIA GPU for Multigrid...")

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

  grid_size = 1024
  num_levels = min(10, grid_size.bit_length())
  model = MyMachine(nt=10, num_levels=num_levels, datatype=torch.float16).to(device).eval()

  print("[INFO] Creating warmup Multigrid inputs (1024x1024, fp16)...")
  coordinates = torch.linspace(0, 1, grid_size, dtype=torch.float16, device=device)
  X, Y = torch.meshgrid(coordinates, coordinates, indexing="ij")

  masks = []
  level_size = grid_size
  for _ in range(10):
    mask = torch.ones((1, 1, level_size, level_size), dtype=torch.float16, device=device)
    mask[:, :, 0, :] = 0
    mask[:, :, -1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, -1] = 0
    masks.append(mask)
    level_size = max(1, level_size // 2)

  model_inputs = (X, Y, *masks)

  print(f"[INFO] Running {warmup_runs} warmup iterations...")
  with torch.inference_mode():
    for _ in range(warmup_runs):
      _ = model(*model_inputs)
  torch.cuda.synchronize(device)

  print("+++ Warmup Done +++")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Warm up the NVIDIA GPU before running Multigrid benchmarks."
  )
  parser.add_argument(
      "--runs",
      type=int,
      default=50,
      help="number of warmup iterations (default: 50)",
  )

  args = parser.parse_args()
  main(warmup_runs=args.runs)
