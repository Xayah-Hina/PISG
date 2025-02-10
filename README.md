# Source code for PISG 2025

## Dependencies:

- core
  - Python 3.11
  - torch 2.6.0 + cu124
  - numpy 2.1.2
  - scipy 1.15.1
  - tqdm 4.67.1
  - pytorch-memlab 0.3.0
- load dataset only
  - imageio 2.37.0
  - av 14.1.0
  - opencv-python 4.11.0.86
- unit tests only
  - memory_profiler 0.61.0
  - matplotlib 3.10.0
- for hyfluid only
  - taichi 1.7.3

```shell
python -m pip uninstall phiflow
python -m pip install git+https://github.com/tum-pbs/PhiFlow@develop
python -m pip uninstall phiml
python -m pip install git+https://github.com/tum-pbs/PhiML@develop
```