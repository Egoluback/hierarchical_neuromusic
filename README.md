# Hierarchical Music Transformer 

## Project description

[Presentation about Hierarchical Transformer(in russian)](https://docs.google.com/presentation/d/1DcA-sDbcwSP1stT-Zc5jt3T3nxezFi2h3lZkY3_SvwM/edit?usp=sharing)

This project provides implementation of [Hierarchical Transformer(Hourglass)](https://arxiv.org/pdf/2110.13711) architecture for symbolic music generation. [Music Transformer](https://arxiv.org/pdf/1809.04281) model with [RPR self-attention](https://arxiv.org/pdf/1803.02155) is used as a basic transformer; it's implementation/training code was forked from [Dmitrii Uspenskii project](https://github.com/wwwwwert/Neuromusic). <br />

There were impemented various types of downsample/upsample functions from the paper and losses for different experiments. In all experiments model was training on [Los Angeles MIDI Dataset 3.1](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset) with REMI Tokenizer. <br />

You can explore all the experiments on the [WandB report](https://api.wandb.ai/links/glinkamusic-ai/q9cq3gfg).

Downsample/upsample functions:
- Naive(reduce with mean/copy)
- Linear(linear layer)
- Attention resampling(attn function with naive/linear function)

Losses:
- CrossEntropyLoss
- CosineCrossEntropyLoss: CELoss + C*cosine distances of consecutive pairs, where C is a constant. The goal is to make them different.
- ExpCosineCrossEntropyLoss: CELoss + C*exp(-cosine distances of all pairs).sum().mean(), where C is a constant. The goal is to encourage similarity to something already known

## Project structure
- **/scripts** - project scripts
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test

## Installation guide

It is strongly recommended to use new virtual environment for this project. Project was developed with Python3.9 and Ubuntu 22.04.2 LTS.

To install all required dependencies and final model run:
```shell
./install_dependencies.sh
```

## Reproduce results
To run train _Music Transformer_ with _REMI_ tokenizer and _Los Angeles MIDI_ dataset:
```shell
python -m train -c scripts/configs/REMI/train_music_tranformer.json
```

To run test inference with _Los Angeles MIDI_ dataset with 512 prompt tokens and generate 512 tokens:
```
python test.py \
   -c scripts/configs/test_LAMD.json \
   -r best_model/model_best.pth \
   -o test_results_LAMD \
   --prompt_length 512 \
   --continue_length 512 \
   -b 1
```

To test model on a custom dataset you need to put MIDI files in some directory.
To run test with custom dataset in _custom_dataset_ directory:
```
python test.py \
   -c scripts/configs/test_custom.json \
   -r best_model/model_best.pth \
   -o test_results_custom \
   --prompt_length 512 \
   --continue_length 512 \
   -b 1 \
   -t custom_dataset/
```
