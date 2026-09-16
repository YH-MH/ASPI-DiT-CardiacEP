# ASPI-DiT

Official implementation of:

**ASPI-DiT: An Amortized Diffusion-Transformer Framework for Ill-Posed Mechanistic Parameter Inversion**

ASPI-DiT generates candidate sets of ionic-mechanism parameters from cardiac action-potential waveforms and evaluates the candidates by forward simulation with the O'Hara-Rudy dynamic (ORd) model.

## Installation

```bash
git clone https://github.com/YH-MH/ASPI-DiT-CardiacEP.git
cd ASPI-DiT-CardiacEP

conda create -n aspi-dit python=3.10
conda activate aspi-dit
pip install -r requirements.txt
```

## External components

The following external components are not redistributed in this repository. Please obtain them from their original sources and comply with their respective licenses.

### Time-series encoders

- [CoST](https://github.com/salesforce/CoST) is used as the default time-series encoder.
- [ST-MEM](https://github.com/vuno/ST-MEM) is used in the encoder ablation experiment.

After obtaining the required encoder, update its import and checkpoint paths in the corresponding configuration file.

### ORd cardiac model

The simulation data are generated using the O'Hara-Rudy dynamic human ventricular action-potential model:

- [Original ORd publication](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002061)
- [Published model correction](https://journals.plos.org/ploscompbiol/article/comment?id=10.1371/annotation/c9345189-f5f9-46c0-a4cf-d774cd858917)

The ORd simulator implementation is not included in this repository.

## Data

The datasets used in the paper are not included in the initial release. They will be uploaded after further organization and documentation.

After obtaining the data, update the dataset paths in the corresponding files under `configs/`.

## Training

Before training, update the data, encoder, checkpoint, and output paths in the configuration and launcher files.

Train the default model with a pretrained frozen CoST encoder:

```bash
python scripts/run_train_ord.py
```

## Inference

Set the trained-model, data, and output paths in the inference launcher, then run:

```bash
python scripts/run_inference_ord.py
```

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{chen2026aspi,
  title     = {ASPI-DiT: An Amortized Diffusion-Transformer Framework for Ill-Posed Mechanistic Parameter Inversion},
  author    = {Chen, Yuhang and He, Yuanbo and Cao, DingLun and Cui, Jiahao and Li, Yacong and Li, Shuai},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages     = {10669--10680},
  year      = {2026}
}
```

## Third-party code and license

The training framework and several utility modules are adapted from [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), Copyright (c) Microsoft Corporation, under the MIT License. The complete license is included in `THIRD_PARTY_LICENSES/VQ-Diffusion-LICENSE`.
