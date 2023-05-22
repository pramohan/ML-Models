import numpy as np
from datasets import *
from random_control import *
from losses import *


class Layer:
    def __init__(self, num_neurons):
        '''
        Set the number of neurons in the layer.
        '''
        self.num_neurons = num_neurons

    def softmax(self, inputs):
        # Return the softmax of the input
        return np.exp(inputs) / np.sum(np.exp(inputs), axis = 1, keepdims=True)

    def tanH(self, inputs):
        # Return the tanh of the input
        return np.tanh(inputs)

    def sigmoid(self, inputs):
        # Return the sigmoid of the input
        return 1 / (1 + np.exp(-inputs))

    def relu(self, inputs):
        # Return the relu of the input.
        return np.maximum(0, inputs)

    def tanH_derivative(self, Z):
        # Calculate the derivative of the tanh function on an input.
        return 1 - np.power(np.tanh(Z), 2)

    def sigmoid_derivative(self, Z):
        # Calculate the derivative of the sigmoid function on an input.
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def relu_derivative(self, Z):
        # Calculate the derivative of the relu function on an input.
        x = Z.copy()
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def forward(self, inputs, weights, bias, activation):
        Z_curr = np.matmul(inputs, weights.T) + bias  # compute Z_curr from weights and bias

        if activation == 'relu':
            A_curr = self.relu(inputs=Z_curr)
        elif activation == 'sigmoid':
            A_curr = self.sigmoid(inputs=Z_curr)
        elif activation == 'tanH':
            A_curr = self.tanH(inputs=Z_curr)
        elif activation == 'softmax':
            A_curr = self.softmax(inputs=Z_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        # We will denote the partial derivative of the loss with respect to each variable as dZ, dW, db, dA
        '''
        The inputs to this function are:
            dA_curr - the partial derivative of the loss with respect to the activation of the preceding layer (l + 1).
            W_curr - the weights of the layer (l)
            Z_curr - the weighted sum of layer (l)
            A_prev - the activation of this layer (l) ... we use prev with respect to dA_curr

        The outputs are the partial derivatives with respect
            dA - the activation of this layer (l) -- needed to continue the backprop
            dW - the weights -- needed to update the weights
            db - the bias -- (needed to update the bias
        '''
        if activation == 'softmax':
            # dA_curr = dZ for this one.
            dW = np.matmul(A_prev.T, dA_curr)
            db = np.sum(dA_curr, axis = 0, keepdims = True)
            dA = np.matmul(dA_curr, W_curr)
        elif activation == 'sigmoid':
            # Computing dZ is not technically needed, but it can be used to help compute the other values.
            activation_derivative = self.sigmoid_derivative(Z_curr)
            dZ = np.multiply(dA_curr, activation_derivative)
            dW = np.matmul(A_prev.T, dZ)
            db = np.sum(dZ, axis = 0, keepdims = True)
            dA = np.matmul(dZ, W_curr)
        elif activation == 'tanH':
            activation_derivative = self.tanH_derivative(Z_curr)
            dZ = np.multiply(dA_curr, activation_derivative)
            dW = np.matmul(A_prev.T, dZ)
            db = np.sum(dZ, axis = 0, keepdims = True)
            dA = np.matmul(dZ, W_curr)
        elif activation == 'relu':
            activation_derivative = self.relu_derivative(Z_curr)
            dZ = np.multiply(dA_curr, activation_derivative)
            dW = np.matmul(A_prev.T, dZ)
            db = np.sum(dZ, axis = 0, keepdims = True)
            dA = np.matmul(dZ, W_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return dA, dW, db

'''
* `MLP` is a class that represents the multi-layer perceptron with a variable number of hidden layer. 
   The constructor initializes the weights and biases for the hidden and output layers.
* `sigmoid`, `relu`, `tanh`, and `softmax` are activation function used in the MLP. 
   They should each map any real value to a value between 0 and 1.
* `forward` computes the forward pass of the MLP. 
   It takes an input X and returns the output of the MLP.
* `sigmoid_derivative`, `relu_derivative`, `tanH_derivative` are the derivatives of the activation functions. 
   They are used in the backpropagation algorithm to compute the gradients.
*  `mse_loss`, `hinge_loss`, `cross_entropy_loss` are each loss functions.
   The MLP algorithms optimizes to minimize those.
* `backward` computes the backward pass of the MLP. It takes the input X, the true labels y, 
   the predicted labels y_hat, and the learning rate as inputs. 
   It computes the gradients and updates the weights and biases of the MLP.
* `train` trains the MLP on the input X and true labels y. It takes the number of epochs 
'''

class MLP:
    def __init__(self, layer_list):
        '''
        Arguments
        --------------------------------------------------------
        layer_list: a list of numbers that specify the width of the hidden layers. 
               The dataset dimensionality (input layer) and output layer (1) 
               should not be specified.
        '''
        self.layer_list = layer_list
        self.network = []  ## layers
        self.architecture = []  ## mapping input neurons --> output neurons
        self.params = []  ## W, b
        self.memory = []  ## Z, A
        self.gradients = []  ## dW, db
        self.loss = []
        self.accuracy = []

        self.loss_func = None
        self.loss_derivative = None

        self.init_from_layer_list(self.layer_list)

    def init_from_layer_list(self, layer_list):
        for layer_size in layer_list:
            self.add(Layer(layer_size))

    def add(self, layer):
        self.network.append(layer)

    def _compile(self, data, activation_func='relu'):
        self.architecture = [] 
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append({'input_dim': data.shape[1], 'output_dim': self.network[idx].num_neurons,
                                          'activation': activation_func})
            elif idx > 0 and idx < len(self.network) - 1:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': activation_func})
            else:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': 'softmax'})
        return self

    def _init_weights(self, data, activation_func, seed=None):
        self.params = []
        self._compile(data, activation_func)

        if seed is None:
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})
        else:
            # For testing purposes
            fixed_generator = np.random.default_rng(seed=seed)
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': fixed_generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})

        return self

    def forward(self, data):
        A_prev = data
        A_curr = None
        
        mem_dict = {'Z': None, 'A': A_prev}
        self.memory.append(mem_dict)

        for i in range(len(self.params)):
            # Compute the forward for each layer and store the appropriate values in the memory.
            # We format our memory_list as a list of dicts
            # mem_dict = {'?': ?}; self.memory.append(mem_dict)
            weight = self.params[i]['W']
            bias = self.params[i]['b']
            activation = self.architecture[i]['activation']
            layer = self.network[i]
            
            A_curr, Z_curr = layer.forward(A_prev, weight, bias, activation)
            A_prev = A_curr

            mem_dict = {'Z': Z_curr, 'A': A_prev}
            
            self.memory.append(mem_dict)

        return A_curr

    def backward(self, predicted, actual):
        ## compute the gradient on predictions
        dscores = self.loss_derivative(predicted, actual)
        dA_prev = dscores  # This is the derivative of the loss function with respect to the output of the last layer

        # Compute the backward for each layer and store the appropriate values in the gradients.
        # We format our gradients_list as a list of dicts
        for i in range(len(self.params) - 1, -1, -1):
            W_curr = self.params[i]['W']
            Z_curr = self.memory[i+1]['Z']
            A_prev = self.memory[i]['A']
            activation = self.architecture[i]['activation']
            layer = self.network[i]
            
            dA_prev, dW, db = layer.backward(dA_prev, W_curr, Z_curr, A_prev, activation)
            
            mem_dict = {'dW': dW, 'db': db}
            self.gradients.insert(0, mem_dict)

    def _update(self, lr):
        # Update the network parameters using the gradients and the learning rate.            
        for i in range(len(self.params)):
            Weight = self.params[i]['W']
            b = self.params[i]['b']
            
            dW = self.gradients[i]['dW'].T
            db = self.gradients[i]['db']
            
            Weight = Weight - lr * dW
            b = b - lr * db
            
            self.params[i] = {'W': Weight, 'b': b}

    # Loss and accuracy functions
    def _calculate_accuracy(self, predicted, actual):
        return np.mean(np.argmax(predicted, axis=1) == actual)

    def _calculate_loss(self, predicted, actual):
        return self.loss_func(predicted, actual)

    def _set_loss_function(self, loss_func_name='negative_log_likelihood'):
        if loss_func_name == 'negative_log_likelihood':
            self.loss_func = negative_log_likelihood
            self.loss_derivative = nll_derivative
        elif loss_func_name == 'hinge':
            self.loss_func = hinge
            self.loss_derivative = hinge_derivative
        elif loss_func_name == 'mse':
            self.loss_func = mse
            self.loss_derivative = mse_derivative
        else:
            raise Exception("Loss has not been specified. Abort")

    def get_losses(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

    def train(self, X_train, y_train, epochs=1000, lr=1e-4, batch_size=16, activation_func='relu', loss_func='negative_log_likelihood'):

        self.loss = []
        self.accuracy = []
        self.grad_norm = []
        self._set_loss_function(loss_func)

        # cast to int
        y_train = y_train.astype(int)

        # initialize network weights
        self._init_weights(X_train, activation_func)

        # Calculate number of batches
        num_datapoints = y_train.shape[0]
        num_batches = int(np.ceil(num_datapoints/batch_size))

        # Shuffle the data and iterate over mini-batches for each epoch.
        # We are implementing mini-batch gradient descent.
        for i in range(int(epochs)):

            batch_loss = 0
            batch_acc = 0
            
            rand_idx = list(range(num_datapoints))
            np.random.shuffle(rand_idx)
            for batch_num in range(num_batches - 1):

                X_batch = X_train[rand_idx[batch_num * batch_size: (batch_num + 1) * batch_size], :]
                y_batch = y_train[rand_idx[batch_num * batch_size: (batch_num + 1) * batch_size]]

                self.memory = []
                self.gradients = []

                yhat = self.forward(X_batch) # Compute yhat

                acc = self._calculate_accuracy(yhat, y_batch)  # Compute and update batch acc
                loss = self._calculate_loss(yhat, y_batch)  # Compute and update batch loss
                
                batch_loss += loss
                batch_acc += acc

                # Stop training if loss is NaN
                if np.isnan(loss) or np.isinf(loss):
                    if len(self.accuracy) == 0:
                        self.accuracy.append(batch_acc / batch_num)
                        self.loss.append(batch_loss / batch_num)
                    s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                    print(s)
                    print("Stopping training because loss is NaN")
                    return

                # Update the network
                self.backward(yhat, y_batch)
                self._update(lr)

            self.loss.append(batch_loss / num_batches)
            self.accuracy.append(batch_acc / num_batches)
            
            grads = [gr['dW'] for gr in self.gradients]
            grad_norms = [np.linalg.norm(grad) for grad in grads]
            self.grad_norm.append(grad_norms)

            if i % 20 == 0:
                s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                print(s)

    def predict(self, X, y, loss_func='negative_log_likelihood'):
        #  Predict the loss and accuracy on a val or test set and print the results.
        yhat = self.forward(X)
        
        # y = np.array(y, dtype=np.int32)
        acc = self._calculate_accuracy(yhat, y)
        loss = self._calculate_loss(yhat, y)
        
        if np.isnan(loss) or np.isinf(loss):
            print("Prediction has NaN/inf loss")

        # for plotting purposes
        self.test_loss = loss  # loss
        self.test_accuracy = acc  # accuracy
