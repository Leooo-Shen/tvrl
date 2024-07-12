# Spatiotemporal Representation Learning for Short and Long Medical Image Time Series (MICCAI 2024)
Implmentation of "[Spatiotemporal Representation Learning for Short and Long Medical Image Time Series](https://arxiv.org/abs/2403.07513)" accepted at MICCAI 2024.



## Environment Setup
```
conda create -n tvrl python=3.9 -y
conda activate tvrl
pip install -r requirements.txt
```

## Usage
### Data Preparation
Prepare your own dataset


### Pretraining
The entry point for pretraining is `pretrain.py`. You can run the following command to pretrain the model.

Note that you will need to adjust the configuration file `config/pretrain.yaml`.
```
```




### Linear Evaluation
The entry point for linear evaluation is `finetune.py`. By default, the script loads the same checkpoint and runs with 5 random seeds to report the average performance with standard deviation.

Note that you will need to adjust the configuration file `config/finetune.yaml`.

```
```

### Hyperparameter Tunning with Wandb Sweep
Wandb sweep can be used to run multiple experiments with different hyperparameters. Please check the [official documentation](https://docs.wandb.ai/guides/sweeps) for more details. We also provide a sample sweep configuration file `sweep/sweep.yaml`.
