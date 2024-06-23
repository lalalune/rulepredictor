# Rule Predictor

Can a transformer predict rules? Maybe!

This package enables generation of simple strings of binary data which have had a rule applied to them. The goal of the transformer is to igure out what the rule is.

First, generate some training data
```
python -m bit_predictor.generate_data
```

Then, train the model
```
python -m bit_predictor.train
```

Finally, test the model
```
python -m bit_predictor.eval
```