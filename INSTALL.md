# Installation Guide

This project is designed to run on both local machines and HPC clusters.
Because clusters vary significantly, installation is split into steps.

# 1. Create Python Environment

Recommended:

```
conda create -n LoDAdaC python=3.9
conda activate LoDAdaC
```

# 2. Install Core Dependencies
```
conda install --file requirements.txt
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
# use system linker and libs
export LD=/usr/bin/ld
export LD_LIBRARY_PATH=/usr/lib64:/lib64:$LD_LIBRARY_PATH
pip install --no-binary=mpi4py mpi4py
```
# 4. Install PyTorch

PyTorch must match your hardware.

CPU only:
```
pip install torch torchvision
```
GPU (example: CUDA 10.2):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu102
```
## Notes
Don't install PyTorch via conda if using system CUDA modules, match CUDA version to cluster environment