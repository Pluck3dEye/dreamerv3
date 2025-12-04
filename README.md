# DreamerV3 for Highway Environments

This project extends the [DreamerV3 implementation by Danijar Hafner](https://github.com/danijar/dreamerv3) with support for [highway-env](https://github.com/Farama-Foundation/HighwayEnv) environments, Windows compatibility, and automation scripts.

DreamerV3 is a scalable and general reinforcement learning algorithm that masters a wide range of applications with fixed hyperparameters.

## Features

- ✅ **Highway-env integration** - Merge, Highway, Roundabout, Parking, Intersection, Racetrack environments
- ✅ **Windows support** - Fixed path handling issues for Windows compatibility
- ✅ **Video recording** - Automatic video capture during evaluation
- ✅ **Automation script** - PowerShell script for easy training and evaluation
- ✅ **Speed configurations** - Configurable vehicle speeds for merge environment
- ✅ **WSL2 GPU support** - Instructions for GPU training on Windows via WSL2

## About DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor critic policy from imagined trajectories. The world model encodes sensory inputs into categorical representations and predicts future representations and rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

To learn more about DreamerV3:
- [Research paper][paper]
- [Project website][website]
- [Original repository][dreamerv3-repo]

# Instructions

The code has been tested on Linux, Mac, and Windows and requires Python 3.11+.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
   - [Windows Setup](#windows-setup)
   - [Linux/Mac Setup](#linuxmac-setup)
   - [WSL2 GPU Setup (Windows)](#wsl2-gpu-setup-windows)
3. [Training](#training)
   - [Basic Training](#basic-training)
   - [Highway Environment Training](#highway-environment-training)
   - [Model Sizes](#model-sizes)
4. [Evaluation](#evaluation)
   - [Video Recording](#video-recording)
5. [Automation Script](#automation-script-train_and_evalps1)
   - [Parameters Reference](#parameters-reference)
   - [Usage Examples](#usage-examples)
6. [Highway Environments](#highway-environments)
   - [Available Environments](#available-environments)
   - [Speed Configurations](#speed-configurations)
7. [Viewing Results](#viewing-results)
8. [Troubleshooting](#troubleshooting)
9. [Tips](#tips)
10. [Disclaimer](#disclaimer)

---

## Quick Start

**Windows PowerShell (CPU):**
```powershell
# Install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.windows.txt

# Train on merge environment for 1 hour
.\train_and_eval.ps1 -Environment merge -TrainDuration 3600

# Or run with video recording
.\train_and_eval.ps1 -Environment merge -SpeedConfig fast -OpenVideoFolder
```

**Linux/Mac (GPU):**
```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train on merge environment
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/train_$(date +%s) \
  --configs highway_merge_rgb size12m \
  --run.duration 3600
```

---

## Installation

### Windows Setup

Windows requires CPU-only JAX (no CUDA support via pip):

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Windows-specific requirements (CPU JAX)
pip install -r requirements.windows.txt
```

**Requirements file (`requirements.windows.txt`):**
- Uses `jax==0.4.33` (CPU only)
- Includes `gymnasium`, `highway-env`, and all dependencies

### Linux/Mac Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with GPU support
pip install -r requirements.txt
```

For GPU support, ensure you have CUDA installed and use the appropriate JAX version.

### WSL2 GPU Setup (Windows)

For GPU training on Windows, use WSL2:

```bash
# In WSL2 Ubuntu
cd /mnt/e/MyWork/dreamerv3-codex-implement-training-for-dreamerv3-in-merge-environment

# Create virtual environment
python3 -m venv ~/dreamerv3-venv
source ~/dreamerv3-venv/bin/activate

# Install JAX with CUDA support
pip install jax==0.4.26
pip install jaxlib==0.4.26+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install gymnasium highway-env pillow ruamel.yaml msgpack cloudpickle rich

# Train with GPU
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/train_gpu_$(date +%s) \
  --configs highway_merge_rgb size12m \
  --run.duration 3600 \
  --jax.platform cuda \
  --env.gym.render_mode rgb_array
```

---

## Training

### Basic Training

```powershell
# Windows (CPU)
python dreamerv3/main.py `
  --logdir "$env:USERPROFILE\logdir\dreamer\train_$(Get-Date -UFormat %s)" `
  --configs highway_merge_rgb size1m `
  --run.duration 3600 `
  --jax.platform cpu `
  --env.gym.render_mode rgb_array `
  --logger.timer False
```

```bash
# Linux/Mac (GPU)
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/train_$(date +%s) \
  --configs highway_merge_rgb size12m \
  --run.duration 3600
```

### Highway Environment Training

Use the `highway_merge_rgb` config for visual observations:

```powershell
python dreamerv3/main.py `
  --logdir "$env:USERPROFILE\logdir\dreamer\merge_train" `
  --configs highway_merge_rgb size1m `
  --run.duration 3600 `
  --jax.platform cpu
```

### Model Sizes

Available model sizes (from smallest to largest):

| Config | Parameters | VRAM | Use Case |
|--------|------------|------|----------|
| `size1m` | ~1M | ~1GB | CPU training, debugging |
| `size4m` | ~4M | ~2GB | Quick experiments |
| `size12m` | ~12M | ~3GB | GPU training |
| `size25m` | ~25M | ~4GB | Good results |
| `size50m` | ~50M | ~6GB | Better results |
| `size100m` | ~100M | ~8GB | High quality |
| `size200m` | ~200M | ~12GB | Best quality |
| `size400m` | ~400M | ~20GB | Maximum quality |

**Important:** Always use the same model size for training and evaluation!

---

## Evaluation

Run evaluation on a trained checkpoint:

```powershell
python dreamerv3/main.py `
  --logdir "$env:USERPROFILE\logdir\dreamer\eval_$(Get-Date -UFormat %s)" `
  --configs highway_merge_rgb size1m `
  --run eval_only `
  --run.from_checkpoint "path\to\checkpoint.ckpt" `
  --run.steps 10000 `
  --jax.platform cpu `
  --env.gym.render_mode rgb_array
```

### Video Recording

Enable video recording during evaluation:

```powershell
python dreamerv3/main.py `
  --logdir "$env:USERPROFILE\logdir\dreamer\eval_video" `
  --configs highway_merge_rgb size1m `
  --run eval_only `
  --run.from_checkpoint "path\to\checkpoint.ckpt" `
  --run.steps 10000 `
  --jax.platform cpu `
  --env.gym.record_video True `
  --env.gym.video_dir "path\to\video_output" `
  --env.amount 1
```

**Video settings:**
- Videos are saved as MP4 at 15 FPS
- One video per evaluation episode
- Use `--env.amount 1` for sequential episodes (required for video recording)

---

## Automation Script (train_and_eval.ps1)

The PowerShell script `train_and_eval.ps1` automates training and evaluation with video recording.

### Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-TrainDuration` | int | 3600 | Training duration in seconds |
| `-ModelSize` | string | "size1m" | Model size config |
| `-EvalSteps` | int | 10000 | Evaluation steps |
| `-EvalEnvs` | int | 1 | Number of evaluation environments |
| `-TrainEnvs` | int | 4 | Number of training environments |
| `-Platform` | string | "cpu" | JAX platform (cpu/cuda) |
| `-TrainRatio` | int | 32 | Training ratio |
| `-BatchSize` | int | 4 | Batch size |
| `-Checkpoint` | string | "" | Resume from checkpoint |
| `-RunName` | string | auto | Custom run name |
| `-LogDir` | string | ~/logdir/dreamer | Log directory |
| `-SpeedConfig` | string | "default" | Speed config (default/fast/veryfast) |
| `-Environment` | string | "merge" | Environment (merge/highway/roundabout/parking/intersection/racetrack) |
| `-SkipTraining` | switch | false | Skip training, eval only |
| `-SkipEval` | switch | false | Skip evaluation |
| `-OpenVideoFolder` | switch | false | Open video folder after eval |
| `-ShowVisual` | switch | false | Show visual window during training |

### Usage Examples

```powershell
# Basic training + evaluation
.\train_and_eval.ps1 -Environment merge -TrainDuration 3600

# Fast speed configuration
.\train_and_eval.ps1 -Environment merge -SpeedConfig fast -TrainDuration 1800

# Use larger model on GPU (WSL2)
.\train_and_eval.ps1 -ModelSize size12m -Platform cuda -TrainDuration 7200

# Skip training, evaluate existing checkpoint
.\train_and_eval.ps1 -SkipTraining -Checkpoint "path\to\checkpoint.ckpt" -OpenVideoFolder

# Train on highway environment with visual feedback
.\train_and_eval.ps1 -Environment highway -ShowVisual -TrainDuration 3600

# Custom run name and log directory
.\train_and_eval.ps1 -RunName "experiment_01" -LogDir "D:\logs\dreamer"

# All environments supported
.\train_and_eval.ps1 -Environment roundabout
.\train_and_eval.ps1 -Environment parking
.\train_and_eval.ps1 -Environment intersection
.\train_and_eval.ps1 -Environment racetrack
```

---

## Highway Environments

### Available Environments

| Environment | Config | Gymnasium ID | Description |
|-------------|--------|--------------|-------------|
| Merge | `highway_merge_rgb` | `merge-v0` | Merge onto highway |
| Highway | `highway_driving` | `highway-v0` | Lane-keeping on highway |
| Roundabout | `highway_roundabout` | `roundabout-v0` | Navigate roundabout |
| Parking | `highway_parking` | `parking-v0` | Park in a slot |
| Intersection | `highway_intersection` | `intersection-v0` | Cross intersection |
| Racetrack | `highway_racetrack` | `racetrack-v0` | Drive on racetrack |

### Speed Configurations

For merge environment, use speed variants:

| Config | Ego Speed | Other Vehicles | Merging Vehicle |
|--------|-----------|----------------|-----------------|
| `highway_merge_rgb` | 30 m/s | 25-35 m/s | 25 m/s |
| `highway_merge_fast` | 40 m/s | 35-45 m/s | 35 m/s |
| `highway_merge_veryfast` | 50 m/s | 45-55 m/s | 45 m/s |

Or use the custom `FastMergeEnv` with `fast-merge-v0`:

```powershell
python dreamerv3/main.py `
  --configs highway_merge_fast size1m `
  --env.gym.ego_speed 45 `
  --env.gym.speed_multiplier 1.5 `
  --jax.platform cpu
```

---

## Viewing Results

### Scope Viewer

```bash
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

### JSONL Metrics

Metrics are saved as JSONL files in the log directory:
- `metrics.jsonl` - Training metrics
- `eval_metrics.jsonl` - Evaluation metrics

### Video Output

When video recording is enabled:
- Videos saved in `{logdir}/eval_*/video/` directory
- Format: MP4 at 15 FPS
- One video per episode

---

## Troubleshooting

### Common Issues

**1. JAX Platform Error (no CUDA)**
```
RuntimeError: Unknown backend: 'cuda' or 'gpu' requested, but no GPU backend was found.
```
**Solution:** Use `--jax.platform cpu` on Windows or set up WSL2 for GPU.

**2. MemoryError with Large Models**
```
MemoryError: Unable to allocate array
```
**Solution:** Use smaller model size (e.g., `size1m` or `size4m`).

**3. Checkpoint Size Mismatch**
```
ValueError: Checkpoint has different shapes than the model
```
**Solution:** Use the same `--configs` (especially model size) for training and evaluation.

**4. JAX Profiler Freeze**
```
WARNING: Profiler blocked main thread for ~90 seconds
```
**Solution:** Add `--logger.timer False` to disable the profiler.

**5. render_mode Error**
```
ValueError: render_mode 'none' not supported
```
**Solution:** Use `--env.gym.render_mode rgb_array` for highway-env.

**6. Checkpoint Not Saving (Windows)**
This was a bug with Windows path handling. It has been fixed in `dreamerv3/main.py`.

**7. Too Many Leaves for PyTreeDef**
```
Error: Too many leaves for PyTreeDef
```
**Solution:** You're loading a checkpoint incompatible with current config. Check your `--logdir` path.

### Performance Tips

- **CPU Training:** Use `size1m` model, expect ~10 steps/second
- **GPU Training (WSL2):** Use `size12m` or larger, expect ~100+ steps/second
- **Video Recording:** Use `--env.amount 1` for proper video capture
- **Long Training:** Use `--logger.timer False` to avoid profiler freezes

---

## Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

---

# Credits

This project is built on top of the [DreamerV3 implementation by Danijar Hafner](https://github.com/danijar/dreamerv3). We extend our sincere gratitude to the original authors for their excellent work and for making the code publicly available.

**Original DreamerV3 Authors:**
- Danijar Hafner
- Jurgis Pasukonis
- Jimmy Ba
- Timothy Lillicrap

---

# Disclaimer

This repository is a fork/extension of the [DreamerV3 reimplementation](https://github.com/danijar/dreamerv3) with added support for highway-env environments, Windows compatibility fixes, and automation scripts. It is unrelated to Google or DeepMind. The original implementation has been tested to reproduce the official results on a range of environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
[dreamerv3-repo]: https://github.com/danijar/dreamerv3
