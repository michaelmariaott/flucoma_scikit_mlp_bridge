import json
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def read_dataset_from_file(file):
  '''
  reads a fluid dataset dict from file and converts it to a numpy array
  '''
  with open(file) as f:
    data = json.load(f)

  input_fft = []

  for i in data['data']:
    input_fft.append(data['data'][i])

  return np.array(input_fft)

def write_model_to_file(model, file):

  with open(file, 'w') as f:
    json.dump(model, f, indent=4)

def mlp_to_fluid_json_dict(mlp):
    '''
    by tedmoore:
    https://github.com/tedmoore/FluCoMa-stuff/blob/master/sklearn_mlp_to_fluid_mlp.py

    "layers":{ // layers are in feedforward order
        "activation": // int from FluidMLPRegressor //,
        "biases": biases for the layer (array of "cols" length)
        "cols": int num of neurons in this layer ,
        "rows": num inputs to this layer,
        "weights: array of "rows" arrays with "cols" items in each,
    }
    '''
    weights = mlp.coefs_
    biases = mlp.intercepts_
    json_dict = {"layers":[{} for _ in range(mlp.n_layers_ - 1)]}
    
    activation_map = {'identity':0,  'logistic':1,  'relu':2, 'tanh':3}
    activation = activation_map[mlp.activation]

    for i, biases_array in enumerate(biases):
        if i == (len(biases) - 1):
            json_dict["layers"][i]["activation"] = 0 # identity for the last layer
        else:
            json_dict["layers"][i]["activation"] = activation
        
        json_dict["layers"][i]["biases"] = list(biases_array)
        json_dict["layers"][i]["cols"] = len(biases_array)
        json_dict["layers"][i]["rows"] = len(weights[i])
        json_dict["layers"][i]["weights"] = list([list(w) for w in weights[i]])
    return json_dict

if __name__ == '__main__':

  parser = argparse.ArgumentParser(
  prog='train',
  description='Train MLP from flucoma datasets'
  )

  parser.add_argument('X')
  parser.add_argument('y')
  parser.add_argument('-o', '--output')

  args = parser.parse_args()

  print('Reading ', args.X, args.y, 'from disc...')
  X = read_dataset_from_file(args.X)
  y = read_dataset_from_file(args.y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
  mlp = MLPRegressor(
    random_state=1,
    max_iter=5000, 
    activation='tanh', 
    solver='adam',
    verbose=True
  )
  mlp.fit(X_train, y_train)
  fluid_json = mlp_to_fluid_json_dict(mlp)
  write_model_to_file(fluid_json, f'{args.output}.json')

  print('Score: ', mlp.score(X_test, y_test))

