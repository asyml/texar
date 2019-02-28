
This directory contains several utilities for, e.g., data pre-processing. 

Instructions of using BPE and WPM encoding are as follows. 
See [examples/transformer](https://github.com/asyml/texar/tree/master/examples/transformer)
for a real example of using these encodings.

**Note that** there are a few different (sub-)word encoding approaches and implementations which are used by several popular models. For example:

* **BPE by Rico Sennrich**: Used in [Transformer](https://github.com/asyml/texar/tree/master/examples/transformer) for machine translation. This is the version in this folder, including both BPE training and encoding/decoding. 
* **BPE by OpenAI**: Used in [GPT-2]() language model. Includes BPE encoding/decoding and provided BPE vocab (no training).
* **BPE by WordPiece**: Used in [BERT](https://github.com/asyml/texar/tree/master/examples/bert) for text embedding. Includes BPE encoding/decoding and provided BPE vocab (no training).
* **SPM by sentencepiece**: Used in [Transformer](https://github.com/asyml/texar/tree/master/examples/transformer) for machine translation. This is the version in this folder, including both SPM training and encoding/decoding. 

### *[Byte Pair Encoding (BPE)](https://arxiv.org/abs/1508.07909)* pipeline

* Add `bin` directory to `PATH` env variable
```bash
TEXAR=$(pwd)  
export PATH=$PATH:$TEXAR/bin
```

* Learning BPE vocab on source and target combined
```bash
cat train.src train.trg | learn_bpe -s 32000 > bpe-codes.32000
```

* Applying BPE on source and target files
```bash
apply_bpe -c bpe-codes.32000 < train.src > train.src.bpe
apply_bpe -c bpe-codes.32000 < train.trg > train.trg.bpe
apply_bpe -c bpe-codes.32000 < dev.src > dev.src.bpe
apply_bpe -c bpe-codes.32000 < dev.trg > dev.trg.bpe
apply_bpe -c bpe-codes.32000 < test.src > test.src.bpe
```

* BPE decoding target to match with references
```bash
mv test.out test.out.bpe
cat test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > test.out
```

##### Evaluate Using Transformer's BLEU tool
```bash
python [TEXAR]/examples/transformer/bleu_tool.py --translation=test.out --reference=test.tgt
```

### Word Piece Model (WPM) pipeline

* This requires installation of the [sentencepiece](https://github.com/google/sentencepiece#python-module) library
```bash
pip install sentencepiece
```
* Learning Word Piece on source and target combined
```bash
spm_train --input=train.src,train.tgt --vocab_size 32000 --model_prefix=wpm-codes
```

* Applying Word Piece on source and target
```bash
spm_encode --model wpm-codes.model --output_format=id < train.src > train.src.wpm
spm_encode --model wpm-codes.model --output_format=id < train.tgt > train.tgt.wpm
spm_encode --model wpm-codes.model --output_format=id < valid.src > valid.src.wpm
spm_encode --model wpm-codes.model --output_format=id < valid.tgt > valid.tgt.wpm
spm_encode --model wpm-codes.model --output_format=id < test.src > test.src.wpm
```

* WPM decoding/detokenising target to match with references
```bash
mv test.out test.wpm
spm_decode --model wpm-codes.model --input_format=id < test.out.wpm > test.out
```
