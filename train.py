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

def parse_arguments():
  '''
  parses commandline arguments, similar syntax to fluid.mlpregressor~
  '''
  parser = argparse.ArgumentParser(
  prog='train',
  description='Train MLP from flucoma datasets'
  )

  # inputs and outputs
  parser.add_argument('X', help="Flucoma .json dataset file for the training inputs.")
  parser.add_argument('y', help="Flucoma .json dataset file for the training ouputs.")
  parser.add_argument('output', help=".json file to save the model")

  # Hidden layers
  parser.add_argument('--hidden_layers', nargs='+', type=int, default=[3, 3], help="Number of neurons in hidden layers (default: [3, 3])")
  
  # Activation functions
  parser.add_argument('--activation', type=str, choices=['identity', 'logistic', 'tanh', 'relu'], default='relu', help="Activation function for the hidden layers and output layer")
  parser.add_argument('--output_activation', type=str, choices=['identity', 'logistic', 'tanh', 'relu'], default=None, help="Activation function for the output layer")
  
  # Learning rate
  parser.add_argument('--learn_rate', type=float, default=0.01, help="Learning rate (default: 0.01)")
  
  # Maximum iterations
  parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations for training (default: 1000)")
  
  # Validation settings
  parser.add_argument('--validation', action='store_true', default=False, help="Enable validation during training")
  
  # Batch size
  parser.add_argument('--batch_size', type=int, default=50, help="Batch size for training (default: 50)")
  
  # Momentum
  parser.add_argument('--momentum', type=float, default=0.9, help="Momentum factor for optimization (default: 0.9). Only used with sgd solver.")

  # Solver
  parser.add_argument('--solver', type=str, choices=['lbfgs', 'sgd', 'adam'], default='sgd', help="The solver for weight optimization (default: sgd).")
  

  return parser.parse_args()

if __name__ == '__main__':

  args = parse_arguments()
  

  print('Reading ', args.X, args.y, 'from disc...')
  X = read_dataset_from_file(args.X)
  y = read_dataset_from_file(args.y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
  mlp = MLPRegressor(
    hidden_layer_sizes=args.hidden_layers,
    max_iter=args.max_iter, 
    activation=args.activation, 
    learning_rate_init=args.learn_rate,
    early_stopping=args.validation,
    batch_size=args.batch_size,
    momentum=args.momentum,
    solver=args.solver,
    verbose=True
  )
  if(args.output_activation):
    mlp.out_activation_ = args.output_activation
  mlp.fit(X_train, y_train)
  fluid_json = mlp_to_fluid_json_dict(mlp)
  write_model_to_file(fluid_json, f'{args.output}')

  print('Score: ', mlp.score(X_test, y_test))

