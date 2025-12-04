# BayBE - FP

## Introduction
Experiment Design via Bayesian Optimization based on [BayBE](https://emdgroup.github.io/baybe/0.12.0/).

## How to run
- Install packages in `requirements.txt` (pip is recommended).
- Cache pre-trained models through `python cache_pretrained_model.py`.
- Define the combinations by modifying `param_grid` in `main.py`. There you can also define invalid combinations in `is_valid_combination()`.
- Launch a run through `run.sh` or `main.py`
- Results are saved in `./output`.