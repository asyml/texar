```
pip install -r requirements
python download_glue_data.py
cd bert_released_models
sh download_model.sh
```
If you want to verify the correctness of the BERT model, we gives a simple method here:

run
```
python example_classifier.py --sanity_check=True
```

Remove estimator. Use our settings.

Distributed settings.

example_classifier.py
