# Hierarchical Music Transformer 

## Project description

[Presentation about this project in general(in russian)](https://docs.google.com/presentation/d/1VHXYBQ_0hllOIsx__vtl1KVxg13-n1NKNoh-QT1Fy_0/edit?usp=sharing)

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
- CosineCrossEntropyLoss: the idea is to make pairs after shortening different $$- \sum_{i=1}^{n} y_i\cdot \log \hat{y_i} + \alpha\cdot \frac{1}{N} \sum_{k=1}^{N} \frac{1}{l_k} \sum_{i=1}^{l_k} \langle x_{l_k 2i}, x_{l_k 2i + 1}\rangle$$
- ExpCosineCrossEntropyLoss: the idea is to encourage similarity to something already known for all pairs after shortening/upsampling $$- \sum_{i=1}^{n} y_i\cdot \log \hat{y_i} + \alpha\cdot \frac{1}{N} \sum_{k=1}^{N} \sum_{i=1}^{l_k} \sum_{j=1}^{l_k} e^{-\langle x_{l_k i}, x_{l_k j}\rangle}$$

## Project structure
- **/scripts** - project scripts
     - hierarchical transformer implementation is located in **/scripts/model/Music_Transformer/hierarchical_music_transformer.py**
     - new losses are located in **/scripts/loss/CELossWrapper.py**
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test

## Hierarchical Transformer params
.json config for hierarchical transformer training is located in **/scripts/configs/REMI/train_hierarchical_music_transformer.json**

The most changeable params are:
- input length: the amount of tokens on the first layer = n
- shorten factor: shorten factor s from the paper
- depth: (x, (y, z, y), x) means x layers with n tokens, y layers with n/s tokens, z layers with n/s^2 tokens, ...
- attn resampling: whether to use attn resampling or not
- updown sample type: type of down/up sample layer. now there are only "linear" and "naive"
- save_updown_hidden: if set to true, upsample tokens will not be used for hidden loss calculations (this attribute is needed for CosineCELoss)

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
