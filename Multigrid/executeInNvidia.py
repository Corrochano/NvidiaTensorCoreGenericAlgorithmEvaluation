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
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# GPU power sampling — tries pynvml first, falls back to nvidia-smi
# ---------------------------------------------------------------------------
try:
  import pynvml

  _HAS_NVML = True
except ImportError:
  _HAS_NVML = False


def _nvidia_smi_power_draw():
  """Query instantaneous GPU power via nvidia-smi (fallback). Returns watts or None."""
  try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip().splitlines()[0])
  except Exception:
    return None


class GpuPowerSampler:
  """Background thread that samples GPU power draw at a fixed interval.

  Uses pynvml (NVML C library) when available; otherwise falls back to
  calling ``nvidia-smi`` in a subprocess, which is slower but always
  available on any NVIDIA driver installation.
  """

  def __init__(self, gpu_index=0, interval_ms=50):
    self._use_nvml = False
    self._handle = None
    self._gpu_index = gpu_index

    if _HAS_NVML:
      try:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._use_nvml = True
      except Exception:
        pass  # fall through to nvidia-smi

    if not self._use_nvml:
      if _nvidia_smi_power_draw() is None:
        raise RuntimeError(
            "Neither pynvml nor nvidia-smi could read GPU power. "
            "Install pynvml (pip install pynvml) or verify your NVIDIA driver."
        )
      interval_ms = max(interval_ms, 200)

    self._interval = interval_ms / 1000.0
    self._samples_w = []  # watts
    self._stop_event = threading.Event()
    self._thread = None

  def _read_power_w(self):
    """Read instantaneous GPU power in watts."""
    if self._use_nvml:
      try:
        return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
      except Exception:
        return None
    else:
      return _nvidia_smi_power_draw()

  def start(self):
    self._samples_w.clear()
    self._stop_event.clear()
    self._thread = threading.Thread(target=self._sample_loop, daemon=True)
    self._thread.start()

  def _sample_loop(self):
    while not self._stop_event.is_set():
      w = self._read_power_w()
      if w is not None:
        self._samples_w.append(w)
      self._stop_event.wait(self._interval)

  def stop(self):
    if self._thread is not None:
      self._stop_event.set()
      self._thread.join(timeout=2.0)
      self._thread = None

  @property
  def samples(self):
    return list(self._samples_w)

  @property
  def mean_power_w(self):
    if not self._samples_w:
      return None
    return sum(self._samples_w) / len(self._samples_w)

  @property
  def peak_power_w(self):
    if not self._samples_w:
      return None
    return max(self._samples_w)

  def estimated_energy_j(self, elapsed_seconds):
    """Estimate total GPU energy as mean_power × elapsed_time."""
    mean = self.mean_power_w
    if mean is None:
      return None
    return mean * elapsed_seconds


# ---------------------------------------------------------------------------
# RAPL energy helpers (for CPU-side energy on Intel/AMD hosts)
# ---------------------------------------------------------------------------
def read_rapl_counters():
  powercap_path = Path("/sys/class/powercap")
  domain_paths = [
      *powercap_path.glob("intel-rapl:*"),
      *powercap_path.glob("intel-rapl-mmio:*"),
  ]
  for parent_path in tuple(domain_paths):
    domain_paths.extend(parent_path.glob("intel-rapl:*"))
    domain_paths.extend(parent_path.glob("intel-rapl-mmio:*"))
  domain_paths = list(dict.fromkeys(domain_paths))

  counters = {}
  for domain_path in domain_paths:
    counter_files = [
        domain_path / "name",
        domain_path / "energy_uj",
        domain_path / "max_energy_range_uj",
    ]
    if not all(path.is_file() for path in counter_files):
      continue
    try:
      name = (domain_path / "name").read_text().strip()
      energy_uj = int((domain_path / "energy_uj").read_text().strip())
      max_energy_uj = int(
          (domain_path / "max_energy_range_uj").read_text().strip()
      )
      key = f"{domain_path.name}:{name}"
      counters[key] = {
          "energy_uj": energy_uj,
          "max_energy_range_uj": max_energy_uj,
      }
    except (OSError, ValueError):
      continue
  return counters


def rapl_energy_delta(start_counters, end_counters):
  deltas = {}
  for key, start_info in start_counters.items():
    if key not in end_counters:
      continue
    end_info = end_counters[key]
    delta_uj = end_info["energy_uj"] - start_info["energy_uj"]
    if delta_uj < 0:
      delta_uj += end_info["max_energy_range_uj"]
    deltas[key] = round(delta_uj / 1_000_000.0, 6)
  return deltas


# ---------------------------------------------------------------------------
# PyTorch Multigrid Model
# ---------------------------------------------------------------------------
class MyMachine(nn.Module):

  def __init__(self, nt=100, num_levels=9, datatype=torch.float32):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(grid_size, iterations, runtimes, datatype, exec_mode):
  """exec_mode: 'cpu'         — run Multigrid on the CPU

             'gpu'         — run on CUDA cores (Tensor Cores disabled)
             'tensorcores' — run on CUDA with Tensor Cores enabled
  """
  exec_mode = exec_mode.lower()
  if exec_mode not in ("cpu", "gpu", "tensorcores"):
    print("Error: exec_mode must be 'cpu', 'gpu', or 'tensorcores'.")
    return

  use_cuda = exec_mode in ("gpu", "tensorcores")
  use_tc = exec_mode == "tensorcores"

  # ---- Validate CUDA when needed ----------------------------------------
  if use_cuda:
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA is not available; an NVIDIA GPU is required")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[INFO] GPU: {gpu_name}  (SM {capability[0]}.{capability[1]})")

    if use_tc and capability < (7, 0):
      raise RuntimeError(
          f"GPU compute capability {capability[0]}.{capability[1]} "
          "does not support Tensor Cores"
      )
  else:
    device = torch.device("cpu")
    capability = None
    print(f"[INFO] Running on CPU ({os.cpu_count()} logical cores)")

  # ---- Tensor Core / TF32 configuration ---------------------------------
  if use_cuda:
    if use_tc:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
    else:
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False

  # ---- Dtype setup -------------------------------------------------------
  if datatype == "fp32":
    torch_dtype = torch.float32
    if use_tc:
      if capability is not None and capability < (8, 0):
        raise RuntimeError(
            "FP32 Tensor Core conv (TF32) requires SM 8.0+ (Ampere or newer). "
            "On SM 7.x (Volta/Turing), use 'fp16' for Tensor Cores."
        )
      print("[INFO] TF32 Tensor Core acceleration enabled for FP32 conv")
    else:
      print("[INFO] CUDA cores baseline: Tensor Cores disabled (standard FP32 ALUs)")
  elif datatype == "fp16":
    torch_dtype = torch.float16
    if not use_tc and use_cuda:
      raise RuntimeError(
          "FP16 in 'gpu' mode cannot disable Tensor Cores: cuDNN unconditionally "
          "uses Tensor Cores for FP16 convolutions on modern NVIDIA GPUs. "
          "To benchmark GPU execution WITHOUT Tensor Cores, use datatype='fp32' with exec_mode='gpu'. "
          "To benchmark FP16 WITH Tensor Cores, use exec_mode='tensorcores'."
      )
  elif datatype == "bf16":
    torch_dtype = torch.bfloat16
    if use_cuda and capability is not None and capability < (8, 0):
      raise RuntimeError("BF16 requires SM 8.0+ (Ampere or newer); use fp16")
    if not use_tc and use_cuda:
      raise RuntimeError(
          "BF16 in 'gpu' mode cannot disable Tensor Cores: cuDNN unconditionally "
          "uses Tensor Cores for BF16 convolutions on modern NVIDIA GPUs. "
          "To benchmark GPU execution WITHOUT Tensor Cores, use datatype='fp32' with exec_mode='gpu'. "
          "To benchmark BF16 WITH Tensor Cores, use exec_mode='tensorcores'."
      )
  else:
    print("Error: datatype not available! Use 'fp32', 'fp16', or 'bf16'.")
    return

  print(f"[INFO] Mode: {exec_mode}  |  Dtype: {datatype}  |  Device: {device}")

  num_levels = min(10, grid_size.bit_length())

  # ---- Load TorchScript model or fall back to fresh model ----------------
  ts_path = f"multigrid{grid_size}_model_{datatype}.pt"
  if os.path.isfile(ts_path):
    print(f"[INFO] Loading TorchScript model: {ts_path}")
    try:
      model = torch.jit.load(ts_path, map_location=device)
    except Exception as e:
      print(
          f"[WARN] Failed to load TorchScript model ({e}); using inline model"
      )
      model = MyMachine(iterations, num_levels, torch_dtype).to(device)
  else:
    print(f"[INFO] TorchScript model not found ({ts_path}); using inline model")
    model = MyMachine(iterations, num_levels, torch_dtype).to(device)
  model.eval()

  # ---- Create Multigrid inputs -------------------------------------------
  print(f"[INFO] Creating Multigrid inputs ({grid_size}x{grid_size})...")
  coordinates = torch.linspace(0, 1, grid_size, dtype=torch_dtype, device=device)
  X, Y = torch.meshgrid(coordinates, coordinates, indexing="ij")

  masks = []
  level_size = grid_size
  for _ in range(10):
    mask = torch.ones((1, 1, level_size, level_size), dtype=torch_dtype, device=device)
    mask[:, :, 0, :] = 0
    mask[:, :, -1, :] = 0
    mask[:, :, :, 0] = 0
    mask[:, :, :, -1] = 0
    masks.append(mask)
    level_size = max(1, level_size // 2)

  model_inputs = (X, Y, *masks)

  # ---- GPU power sampler (only when running on CUDA) --------------------
  gpu_sampler = None
  if use_cuda:
    try:
      gpu_sampler = GpuPowerSampler(gpu_index=0, interval_ms=50)
      backend = "pynvml" if gpu_sampler._use_nvml else "nvidia-smi"
      print(f"[INFO] GPU power sampling via {backend}")
    except Exception as exc:
      print(f"[WARN] GPU power sampling unavailable: {exc}")

  # ---- Run inference -----------------------------------------------------
  print(
      f"[INFO] Running {runtimes} run(s) of Multigrid ({iterations} iterations)"
      f" on {exec_mode}..."
  )

  min_time_ms = math.inf

  # CUDA events for GPU timing (only when on CUDA)
  if use_cuda:
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)

  # Start energy/power measurement
  rapl_start = read_rapl_counters()
  if gpu_sampler:
    gpu_sampler.start()
  wall_start = time.perf_counter()

  with torch.inference_mode():
    for _ in range(runtimes):
      if use_cuda:
        cuda_start.record()
        _ = model(*model_inputs)
        cuda_end.record()
        torch.cuda.synchronize(device)
        iter_ms = cuda_start.elapsed_time(cuda_end)
      else:
        iter_start = time.perf_counter()
        _ = model(*model_inputs)
        iter_ms = (time.perf_counter() - iter_start) * 1000.0
      min_time_ms = min(min_time_ms, iter_ms)

  wall_elapsed = time.perf_counter() - wall_start
  if gpu_sampler:
    gpu_sampler.stop()
  rapl_end = read_rapl_counters()

  mean_time_s = wall_elapsed / runtimes
  peak_time_s = min_time_ms / 1000.0  # convert ms → s

  # ---- RAPL CPU energy ---------------------------------------------------
  cpu_energy = rapl_energy_delta(rapl_start, rapl_end)
  cpu_power_mw = None
  if cpu_energy:
    cpu_power_mw = {
        domain: round(joules * 1000 / wall_elapsed, 3)
        for domain, joules in cpu_energy.items()
    }

  # ---- GPU energy --------------------------------------------------------
  gpu_energy_j = None
  gpu_mean_power_w = None
  gpu_peak_power_w = None
  gpu_num_samples = 0
  if gpu_sampler:
    gpu_energy_j = gpu_sampler.estimated_energy_j(wall_elapsed)
    gpu_mean_power_w = gpu_sampler.mean_power_w
    gpu_peak_power_w = gpu_sampler.peak_power_w
    gpu_num_samples = len(gpu_sampler.samples)

  # ---- Console output ----------------------------------------------------
  print("*" * 100)
  print(
      f"Multigrid {grid_size}×{grid_size}  |  {exec_mode}  |  {datatype}  |"
      f"  {iterations} iter  |  {runtimes} run(s)"
  )
  print(f"  Mean wall time  : {mean_time_s:.6f} s")
  print(f"  Peak time       : {peak_time_s:.6f} s  ({min_time_ms:.3f} ms)")
  if cpu_energy:
    print(f"  CPU RAPL energy : {json.dumps(cpu_energy, sort_keys=True)} J")
    print(f"  CPU RAPL power  : {json.dumps(cpu_power_mw, sort_keys=True)} mW")
  if gpu_energy_j is not None:
    print(f"  GPU energy      : {gpu_energy_j:.3f} J  ({gpu_num_samples} samples)")
    print(f"  GPU mean power  : {gpu_mean_power_w:.3f} W")
    print(f"  GPU peak power  : {gpu_peak_power_w:.3f} W")
  print("*" * 100)

  # ---- CSV: timing -------------------------------------------------------
  timing_filename = f"multigrid_{grid_size}x{grid_size}_{exec_mode}.csv"
  if os.path.exists(timing_filename):
    timing_line = f"{datatype};{iterations};{mean_time_s:.6f};{peak_time_s:.6f};\n"
  else:
    timing_line = (
        "datatype;iterations;MeanTime_s;PeakTime_s\n"
        f"{datatype};{iterations};{mean_time_s:.6f};{peak_time_s:.6f};\n"
    )
  with open(timing_filename, "a") as f:
    f.write(timing_line)

  # ---- CSV: CPU energy (RAPL) --------------------------------------------
  if cpu_energy:
    cpu_energy_filename = f"multigrid_cpuEnergy_{grid_size}x{grid_size}_{exec_mode}.csv"
    with open(cpu_energy_filename, "a") as f:
      if f.tell() == 0:
        f.write("datatype;iterations;runtimes;EnergyByDomain_J\n")
      f.write(
          f"{datatype};{iterations};{runtimes};"
          f"{json.dumps(cpu_energy, sort_keys=True)}\n"
      )

    cpu_power_filename = f"multigrid_cpuPower_{grid_size}x{grid_size}_{exec_mode}.csv"
    with open(cpu_power_filename, "a") as f:
      if f.tell() == 0:
        f.write("datatype;iterations;runtimes;AveragePowerByDomain_mW\n")
      f.write(
          f"{datatype};{iterations};{runtimes};"
          f"{json.dumps(cpu_power_mw, sort_keys=True)}\n"
      )

  # ---- CSV: GPU energy (NVML) --------------------------------------------
  if gpu_energy_j is not None:
    gpu_energy_filename = f"multigrid_gpuEnergy_{grid_size}x{grid_size}_{exec_mode}.csv"
    with open(gpu_energy_filename, "a") as f:
      if f.tell() == 0:
        f.write(
            "datatype;iterations;runtimes;gpu_energy_J;"
            "gpu_mean_power_W;gpu_peak_power_W;samples\n"
        )
      f.write(
          f"{datatype};{iterations};{runtimes};"
          f"{gpu_energy_j:.6f};{gpu_mean_power_w:.3f};"
          f"{gpu_peak_power_w:.3f};{gpu_num_samples}\n"
      )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run Multigrid on CPU, GPU (CUDA cores), or Tensor Cores."
  )
  parser.add_argument("grid_input", type=int, help="grid size")
  parser.add_argument(
      "iteration_input",
      type=int,
      help="number of Jacobi iterations in the model",
  )
  parser.add_argument("int_input", type=int, help="number of inference runs")
  parser.add_argument(
      "datatype_input", type=str, help="datatype: fp32/fp16/bf16"
  )
  parser.add_argument(
      "device_input", type=str, help="exec mode: cpu/gpu/tensorcores"
  )

  args = parser.parse_args()
  if args.grid_input <= 0 or args.iteration_input <= 0 or args.int_input <= 0:
    parser.error(
        "grid size, iteration count, and inference run count must be positive"
    )
  main(
      args.grid_input,
      args.iteration_input,
      args.int_input,
      args.datatype_input,
      args.device_input,
  )
