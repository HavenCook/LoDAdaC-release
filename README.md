# LoDAdaC

## Overview
See [publication](./LoDAdaC_anonymized.pdf) for explanation of theory and architecture as well as experimental results on AiMOS. Source code and experiment run scripts both for local, single-node, execution and distributed, multi-node, execution using slurm.

## Repository Structure

```
LoDAdaC-release/
├── config/             # configuration scripts for model setup
├── DOCS/               # further documentation
├── models/             # model definitions
├── src/                # source code
├── scripts/            # job scripts / utilities
├── data/               # input training data 
├── requirements.txt    # core Python dependencies
├── INSTALL.md          # detailed setup instructions
└── README.md
```


## Requirements
- Python ≥ 3.9
- Linux environment (tested on HPC clusters)
- MPI implementation (e.g., OpenMPI)

## Quick Start

```
git clone git@github.com:HavenCook/LoDAdaC-release.git
cd LoDAdaC-release

# Create environment
conda create -n myenv python=3.9
conda activate myenv

# Install core dependencies
pip install -r requirements.txt
```

Then follow INSTALL.md for MPI and PyTorch setup. Once that is done, modify scripts/runscript.sh to load modules and the correct environment.

```
sbatch scripts/runscript.sh
```

Alternatively, for single-node execution, simply call:

```
mpirun -np [np] python3 -u -m scripts.experiment
```