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
    x = torch.matmul(A, B)
    return x


def main(matrix_size, datatype):
  if matrix_size <= 0:
    raise ValueError("matrix_size must be greater than zero")

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
  print("Starting the Tensor Core matrix multiplication test...\n")

  model = MyMachine().to(device).eval()

  if datatype == "fp32":
    torchfloat = torch.float32
    if capability >= (8, 0):
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      print("FP32 matmul enabled with TF32 Tensor Core acceleration (Ampere+)")
    else:
      raise RuntimeError(
          "FP32 Tensor Core matmul (TF32) requires compute capability 8.0+; use fp16"
      )
  elif datatype == "fp16":
    torchfloat = torch.float16
  elif datatype == "bf16":
    if capability >= (8, 0):
      torchfloat = torch.bfloat16
    else:
      raise RuntimeError(
          "BF16 Tensor Core matmul requires compute capability 8.0+; use fp16"
      )
  else:
    print("Error: datatype not available! Use 'fp32', 'fp16', or 'bf16'.")
    return

  print("--------------------------")
  print("Creating matrices...")
  print("--------------------------\n")
  example_A = torch.rand(matrix_size, matrix_size, dtype=torchfloat, device=device)
  example_B = torch.rand(matrix_size, matrix_size, dtype=torchfloat, device=device)

  print("--------------------------")
  print("Testing the PyTorch model on CUDA (Tensor Cores):")
  print("--------------------------\n")
  # Warm-up pass
  with torch.inference_mode():
    _ = model(example_A, example_B)
  torch.cuda.synchronize(device)

  # Timed pass using CUDA events for accurate GPU timing
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  with torch.inference_mode():
    start_event.record()
    output = model(example_A, example_B)
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
  onnx_path = f"matmul{matrix_size}_model_{datatype}.onnx"
  # ONNX export requires CPU tensors as example inputs
  example_A_cpu = example_A.cpu()
  example_B_cpu = example_B.cpu()
  model_cpu = model.cpu()
  torch.onnx.export(
      model_cpu,
      (example_A_cpu, example_B_cpu),
      onnx_path,
      input_names=["A", "B"],
      output_names=["output"],
      dynamic_axes={
          "A": {0: "M", 1: "K"},
          "B": {0: "K", 1: "N"},
          "output": {0: "M", 1: "N"},
      },
      opset_version=17,
  )
  print(f"  Saved ONNX model → {onnx_path}\n")

  # --- Export: TorchScript (alternative for libtorch / Triton Server) --------
  print("--------------------------")
  print("Exporting model to TorchScript...")
  print("--------------------------\n")
  ts_path = f"matmul{matrix_size}_model_{datatype}.pt"
  model_gpu = model.to(device)
  traced_model = torch.jit.trace(model_gpu, (example_A, example_B))
  traced_model.save(ts_path)
  print(f"  Saved TorchScript model → {ts_path}\n")

  print("+++ OK +++")
  print("\nNext steps to optimise for Tensor Cores:")
  print("  1. Convert the ONNX model to a TensorRT engine:")
  print(f"       trtexec --onnx={onnx_path} --fp16 --saveEngine=matmul{matrix_size}.engine")
  print("  2. Or load the TorchScript model in C++/Python with libtorch for deployment.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="python gemmConversion.py matrix_size datatype."
  )

  parser.add_argument(
      "matrix_input", type=int, help="matrix size in integer"
  )
  parser.add_argument(
      "datatype_input", type=str, help="datatype: fp32/fp16/bf16"
  )

  args = parser.parse_args()

  if args.matrix_input is None or args.datatype_input is None:
    print("Error: You must provide all the arguments.")
    parser.print_usage()
  else:
    main(args.matrix_input, args.datatype_input)
