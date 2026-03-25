


# Installation Guide

This project is designed to run on both local machines and HPC clusters.
Because clusters vary significantly, installation is split into steps.

# 1. Create Python Environment

Recommended:

```
conda create -n myenv python=3.9
conda activate myenv
```

# 2. Install Core Dependencies
```
pip install -r requirements.txt
```

These are portable and should work on most systems.

# 3. Install MPI + mpi4py (HPC ONLY)
## Load MPI module
```
module load openmpi
```
(Your cluster may use a different module name.)

## Install mpi4py (IMPORTANT)

AiMOS had trouble with prebuilt mpi4py installations, building from source might be necessary on your cluster:
```
pip install --no-binary=mpi4py mpi4py
```
# 4. Install PyTorch

PyTorch must match your hardware.

CPU only:
```
pip install torch torchvision
```
GPU (example: CUDA 11.8):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
## Notes
Don't install PyTorch via conda if using system CUDA modules, match CUDA version to cluster environment