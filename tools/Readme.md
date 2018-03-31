### *[Byte Pair Encoding](https://arxiv.org/abs/1508.07909)* (BPE) pipeline

* Add `tools` directory to `PATH` env variable
```bash
TXTGEN=$(pwd)  
export PATH=$PATH:$TXTGEN/tools
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

##### Evaluate Using t2t-Bleu
```bash
t2t-bleu --translation=test.out --reference=test.tgt
```

### Word Piece Model (WPM)

* This requires installation of *[sentencepiece](https://github.com/google/sentencepiece#python-module) library
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