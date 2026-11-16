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

  def __init__(self, nt=100, num_levels=9, datatype=torch.float32):
    super(MyMachine, self).__init__()
    self.datatype = datatype
    self.num_levels = int(num_levels)
    self.nt = int(nt)

    # 3x3 5-point discrete Laplace stencil kernel for smoothing
    kernel = torch.tensor(
        [[0.0, 0.25, 0.0],
         [0.25, 0.0, 0.25],
         [0.0, 0.25, 0.0]],
        dtype=self.datatype,
    ).view(1, 1, 3, 3)
    self.register_buffer("kernel", kernel)

    # 2x2 restriction kernel for downsampling
    restriction_kernel = torch.full(
        (1, 1, 2, 2),
        0.25,
        dtype=self.datatype,
    )
    self.register_buffer("restriction_kernel", restriction_kernel)

  def restriction(self, residual):  # Convert fine grid to coarser grid
    return F.conv2d(residual, self.restriction_kernel, stride=2)

  def interpolate(self, f, target_size):  # Convert coarser grid to finer grid
    return F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)

  def jacobi(self, z, mask):  # Jacobi relaxation
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

    # Downward phase
    for level in range(1, num_levels):
      m = masks[level - 1]
      masked_input = grids[level - 1] * m
      unmasked_input = grids[level - 1] * (1.0 - m)

      masked_output = F.conv2d(masked_input, self.kernel, padding=1)
      unmasked_output = F.conv2d(unmasked_input, self.kernel, padding=1)

      residual = masked_output + unmasked_output
      coarse_residual = self.restriction(residual)
      grids.append(coarse_residual)

    # Solve on the coarsest grid
    coarse_solution = grids[-1]
    coarse_solution = self.jacobi(coarse_solution, masks[num_levels - 1])
    grids[-1] = coarse_solution

    # Upward phase
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

    # Final refinement on the finest grid
    final_solution = grids[0]
    final_solution = self.jacobi(final_solution, Mask1)
    return final_solution


def main(grid_size, iterations, datatype):
  if grid_size <= 0:
    raise ValueError("grid_size must be greater than zero")
  if iterations <= 0:
    raise ValueError("iterations must be greater than zero")

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; an NVIDIA GPU is required")

  device = torch.device("cuda")
  capability = torch.cuda.get_device_capability(device)
  if capability < (7, 0):
    raise RuntimeError(
        f"GPU compute capability {capability[0]}.{capability[1]} does not support Tensor Cores"
    )

  gpu_name = torch.cuda.get_device_name(device)
  print(f"GPU: {gpu_name}  (compute capability {capability[0]}.{capability[1]})")
  print("Starting the Tensor Core Multigrid test...\n")

  if datatype == "fp32":
    torchfloat = torch.float32
    if capability >= (8, 0):
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      print("FP32 convolution enabled with TF32 Tensor Core acceleration (Ampere+)")
    else:
      raise RuntimeError(
          "FP32 Tensor Core conv (TF32) requires compute capability 8.0+; use fp16"
      )
  elif datatype == "fp16":
    torchfloat = torch.float16
  elif datatype == "bf16":
    if capability >= (8, 0):
      torchfloat = torch.bfloat16
    else:
      raise RuntimeError(
          "BF16 Tensor Core conv requires compute capability 8.0+; use fp16"
      )
  else:
    print("Error: datatype not available! Use 'fp32', 'fp16', or 'bf16'.")
    return

  num_levels = min(10, grid_size.bit_length())
  model = MyMachine(iterations, num_levels, torchfloat).to(device).eval()

  print("--------------------------")
  print("Creating Multigrid inputs...")
  print("--------------------------\n")
  coordinates = torch.linspace(0, 1, grid_size, dtype=torchfloat, device=device)
  X, Y = torch.meshgrid(coordinates, coordinates, indexing="ij")

  masks = []
  level_size = grid_size
  for _ in range(10):
    mask = torch.ones((1, 1, level_size, level_size), dtype=torchfloat, device=device)
    mask[:, :, 0, :] = 0
    mask[:, :, -1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, -1] = 0
    masks.append(mask)
    level_size = max(1, level_size // 2)

  model_inputs = (X, Y, *masks)

  print("--------------------------")
  print("Testing the PyTorch model on CUDA (Tensor Cores):")
  print("--------------------------\n")
  # Warm-up pass
  with torch.inference_mode():
    _ = model(*model_inputs)
  torch.cuda.synchronize(device)

  # Timed pass using CUDA events for accurate GPU timing
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  with torch.inference_mode():
    start_event.record()
    output = model(*model_inputs)
    end_event.record()

  torch.cuda.synchronize(device)
  elapsed_ms = start_event.elapsed_time(end_event)

  print(f"  Output shape : {tuple(output.shape)}")
  print(f"  Output dtype : {output.dtype}")
  print(f"  Device       : {output.device}")
  print(f"  Elapsed time : {elapsed_ms:.3f} ms\n")

  # --- Export: ONNX (for TensorRT) ------------------------------------------
  print("--------------------------")
  print("Exporting model to ONNX (for TensorRT)...")
  print("--------------------------\n")
  onnx_path = f"multigrid{grid_size}_model_{datatype}.onnx"
  model_cpu = MyMachine(iterations, num_levels, torchfloat).cpu().eval()
  X_cpu = X.cpu()
  Y_cpu = Y.cpu()
  masks_cpu = [m.cpu() for m in masks]
  model_inputs_cpu = (X_cpu, Y_cpu, *masks_cpu)

  input_names = ["X", "Y"] + [f"Mask{i+1}" for i in range(10)]
  dynamic_axes = {
      "X": {0: "H", 1: "W"},
      "Y": {0: "H", 1: "W"},
      "output": {2: "H", 3: "W"},
  }
  for i in range(10):
    dynamic_axes[f"Mask{i+1}"] = {2: f"H{i+1}", 3: f"W{i+1}"}

  torch.onnx.export(
      model_cpu,
      model_inputs_cpu,
      onnx_path,
      input_names=input_names,
      output_names=["output"],
      dynamic_axes=dynamic_axes,
      opset_version=17,
  )
  print(f"  Saved ONNX model → {onnx_path}\n")

  # --- Export: TorchScript (alternative for libtorch / Triton Server) --------
  print("--------------------------")
  print("Exporting model to TorchScript...")
  print("--------------------------\n")
  ts_path = f"multigrid{grid_size}_model_{datatype}.pt"
  model_gpu = model.to(device)
  traced_model = torch.jit.trace(model_gpu, model_inputs)
  traced_model.save(ts_path)
  print(f"  Saved TorchScript model → {ts_path}\n")

  print("+++ OK +++")
  print("\nNext steps to optimise for Tensor Cores:")
  print("  1. Convert the ONNX model to a TensorRT engine:")
  print(f"       trtexec --onnx={onnx_path} --fp16 --saveEngine=multigrid{grid_size}.engine")
  print("  2. Or load the TorchScript model in C++/Python with libtorch for deployment.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="python multigridConversion.py grid_size number_of_iterations datatype."
  )

  parser.add_argument("grid_input", type=int, help="grid size in integer")
  parser.add_argument(
      "iteration_input",
      type=int,
      help="number of Jacobi iterations in the model",
  )
  parser.add_argument(
      "datatype_input", type=str, help="datatype: fp32/fp16/bf16"
  )

  args = parser.parse_args()

  if (
      args.grid_input is None
      or args.iteration_input is None
      or args.datatype_input is None
  ):
    print("Error: You must provide all the arguments.")
    parser.print_usage()
  else:
    main(args.grid_input, args.iteration_input, args.datatype_input)
