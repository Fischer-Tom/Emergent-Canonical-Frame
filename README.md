# Emergent-Canonical-Frame
Official Codebase of the paper *Emergence of a Shared Canonical Object Frame from In-the-Wild Videos*

## Coming Soon
The code is still under construction and will be released soon! 

## Installation

> python3 -m venv .venv
>
> source .venv/bin/activate
>
> python -m pip install --upgrade pip setuptools wheel ninja


> python -m pip install -e . 
> export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

Then install `nvdiffrast`:

> mkdir -p thirdparty && cd thirdparty && git clone https://github.com/NVlabs/nvdiffrast.git && cd nvdiffrast && python -m pip install --no-cache-dir --no-build-isolation . && cd ../..

> cd thirdparty && git clone https://github.com/facebookresearch/uco3d.git && cd uco3d && python -m pip install --no-cache-dir -e . && cd ../..

