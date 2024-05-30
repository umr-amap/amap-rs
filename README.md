# amap-rs

AMAP Remote Sensing Toolbox

This is still a work in progress for local use. Please feel free to open an issue and discuss how to improve this code.

## Installation

install should work via 

```
pip install git+https://github.com/umr-amap/amap-rs.git
```

## Update

for now, update works via reinstall, as releases are done with simple merges from dev branch (i.e. no version number)

```
pip install -U --force-reinstall git+https://github.com/umr-amap/amap-rs.git
```

## Dependencies

Main depencies are `torchgeo`, `timm`, `umap-learn` and `scikit-learn`. All other dependencies come from these.

## Developpement

you can install a local version of the plugin via 

```
pip install -e path/to/this/repo
```
