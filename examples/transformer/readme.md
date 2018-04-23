# Obtain the dataset
```
refer to `https://github.com/shrshore/transformer_pytorch`, which is currently a private repo.
```

# training the model:

```
bash run_en_vi.sh $1

$1=1, for train,
$1=2, for test
```

# evaluation on test dataset newstest2014
```
# enter the args.log_dir directory
cat #model.b5alpha0.6.output.txt | sed 's/@@ //g' > outputs.words
t2t-bleu --translation=outputs.words --reference=newstest2014.tok.de
```
