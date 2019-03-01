#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "You must enter the model name as a parameter, e.g.: sh gpt2_pretrained_models/download_model.sh 117M"
    exit 1
fi

model=$1

mkdir -p gpt2_pretrained_models/$model
for filename in checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe; do
  fetch=$model/$filename
  echo "Fetching $fetch"
  curl --output gpt2_pretrained_models/$fetch https://storage.googleapis.com/gpt-2/models/$fetch
done
