1. obtain the pytorch saved model
```
source ~/.bashrc # add conda path
source activate py36 # there are latest version of tf and pytorch environments
cd /home/hzt/shr/transformer_pytorch
bash tools/bpe_pipeline_en_vi.sh

python resave_devendra_model.py &> logging.txt

```
2. load the pytorch saved model and running test epoch
```
cd /home/hzt/shr/txtgen/examples/transformer
bash run_en_vi 4
```
3. check the output
```
cd /home/hzt/shr/transformer_pytorch/temp/run_en_vi/models/
vim ckpt_from_pytorch.p.test.beam5alpha0.6.outputs.decodes
```


