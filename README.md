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

## Evaluation

Evaluate a checkpoint with the same pose-fitting and reporting pipeline used by the original repository:

    python scripts/eval.py evaluation.checkpoint=/path/to/checkpoint.pt

For a ground-truth-mask diagnostic run:

    python scripts/eval.py evaluation.checkpoint=/path/to/checkpoint.pt evaluation.use_gt_mask=true

By default, evaluation uses the configured UCO3D validation split and writes per-dataset metrics, pose alignments, and a summary CSV below `eval_logs`. Override datasets or dataset paths through the same Hydra options used for training.

ImageNet3D and the other ported single-image evaluation datasets can be selected through the dataset config group:

    python scripts/eval.py dataset=imagenet3d dataset.data_root=/path/to/imagenet3d evaluation.checkpoint=/path/to/checkpoint.pt

Available configs are `imagenet3d`, `pascal3d`, `objectron`, `sunrgbd`,
`arkitscenes`, `omni6dpose`, and `real275`. Their placeholder paths must be
overridden with the corresponding local dataset and CAD roots.
