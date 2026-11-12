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

  def __init__(self, nt=100, datatype=torch.float32):
    super(MyMachine, self).__init__()
    self.datatype = datatype
    self.nt = int(nt)

    # 3x3 5-point discrete Laplace stencil kernel for Jacobi relaxation
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
  print("Starting the Tensor Core Jacobi relaxation test...\n")

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

  model = MyMachine(iterations, torchfloat).to(device).eval()

  print("--------------------------")
  print("Creating Jacobi inputs...")
  print("--------------------------\n")
  coordinates = torch.linspace(0, 1, grid_size, dtype=torchfloat, device=device)
  x_coords, y_coords = torch.meshgrid(coordinates, coordinates, indexing="ij")

  x = torch.exp(
      -50.0 * ((x_coords - 0.5) ** 2 + (y_coords - 0.5) ** 2)
  ).unsqueeze(0).unsqueeze(0)
  x_prev = x.clone()

  mask = torch.ones_like(x)
  mask[:, :, 0, :] = 0
  mask[:, :, -1, :] = 0
  mask[:, :, :, 0] = 0
  mask[:, :, :, -1] = 0

  print("--------------------------")
  print("Testing the PyTorch model on CUDA (Tensor Cores):")
  print("--------------------------\n")
  # Warm-up pass
  with torch.inference_mode():
    _ = model(x, x_prev, mask)
  torch.cuda.synchronize(device)

  # Timed pass using CUDA events for accurate GPU timing
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  with torch.inference_mode():
    start_event.record()
    output = model(x, x_prev, mask)
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
  onnx_path = f"jacobi{grid_size}_model_{datatype}.onnx"
  x_cpu = x.cpu()
  x_prev_cpu = x_prev.cpu()
  mask_cpu = mask.cpu()
  model_cpu = MyMachine(iterations, torchfloat).cpu().eval()

  torch.onnx.export(
      model_cpu,
      (x_cpu, x_prev_cpu, mask_cpu),
      onnx_path,
      input_names=["X", "X_prev", "Mask"],
      output_names=["output"],
      dynamic_axes={
          "X": {2: "H", 3: "W"},
          "X_prev": {2: "H", 3: "W"},
          "Mask": {2: "H", 3: "W"},
          "output": {2: "H", 3: "W"},
      },
      opset_version=17,
  )
  print(f"  Saved ONNX model → {onnx_path}\n")

  # --- Export: TorchScript (alternative for libtorch / Triton Server) --------
  print("--------------------------")
  print("Exporting model to TorchScript...")
  print("--------------------------\n")
  ts_path = f"jacobi{grid_size}_model_{datatype}.pt"
  model_gpu = model.to(device)
  traced_model = torch.jit.trace(model_gpu, (x, x_prev, mask))
  traced_model.save(ts_path)
  print(f"  Saved TorchScript model → {ts_path}\n")

  print("+++ OK +++")
  print("\nNext steps to optimise for Tensor Cores:")
  print("  1. Convert the ONNX model to a TensorRT engine:")
  print(f"       trtexec --onnx={onnx_path} --fp16 --saveEngine=jacobi{grid_size}.engine")
  print("  2. Or load the TorchScript model in C++/Python with libtorch for deployment.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="python jacobiConversion.py grid_size number_of_iterations datatype."
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
