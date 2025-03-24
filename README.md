# flucoma_scikit_mlp_bridge
Train a MLP in scikit-learn based on FluCoMa datasets and convert the trained model back to FluCoMa.

## Usage
Make a virtual environement (the following commands are for macOS):
```
$ python -m pip venv venv
$ source venv/bin/activate
```
Install the requirements:
```
$ python -m pip install -r requirements.txt
```
Export your datasets as json files from FluCoMa. Invoke the python script as follows:
```
$ python train.py X_dataset.json y_dataset.json -o output_name
```
