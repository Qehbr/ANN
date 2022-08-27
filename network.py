### Libraries
import numpy as np
import random
from cost_function import CostFunction
from ReLU import ReLU


# SoftMax function used in output layer (if flag is given)
def softmax(layer):
    temp = np.exp(layer)
    return temp / np.sum(temp, axis=1, keepdims=True)


class Network(object):
    def __init__(self, layer_dims, cost_function, activation_functions=None, use_softmax=False):
        """
        Network initialization function is responsible for proper creation of Network object.
        Parameters:
            ''layer_dims'' - list containing number of neurons in each layer ;
                             example: [784, 50, 20, 10]
            ''cost_function'' - cost_function object (look cost_function.py) ;
                                example: SquaredSum()
            ''activation_functions'' - list of activation functions for each of hidden layers ;
                                       example: [ReLU(), Tanh()]
            ''use_softmax'' - flag whether to use softmax function for output layer ;
                              example: True
        """
        # initialization of variables
        self.use_softmax = use_softmax
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.weights = []
        self.biases = []
        self.cost_function = cost_function
        # activation_functions all ReLU by default
        if activation_functions == None:
            self.activation_functions = [ReLU() for i in range(len(self.layer_dims) - 2)]
        elif len(activation_functions) != len(self.layer_dims) - 2:
            raise ValueError("activation_functions variable should be as number of hidden layers")
        else:
            self.activation_functions = activation_functions

        # random generation of weights and biases
        for i in range(len(layer_dims) - 1):
            # weights are divided by 'np.sqrt(layer_dims[i])' for faster learning
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i + 1]) / np.sqrt(layer_dims[i]))
            self.biases.append(np.random.randn(1, layer_dims[i + 1]))

    def fit(self, training_data, alpha, epochs, mini_batch_size, test_data=None, evaluation_data=None, dropout=None,
            lmbda=0.0, early_stop_epochs=10, early_stop_variance=0.00000001, early_stop_min_alpha=None):
        """
        Fit is learning+testing function of Network variable.
        Parameters:
            ''training_data'' - list of tuples (x,y) where 'x' is np.array input and 'y' is np.array output. Used for training ANN ;
                                example: [([1],[2])([3],[4])]
            ''alpha''(eta) - learning rate of ANN ;
                             example: 0.01
            ''epochs'' - number of epochs (iterations) over training data ;
                         example: 10
            ''mini_batch_size'' - size of mini batch in stochastic gradient descent ; example: 100
            ''test_data'' - list of tuples (x,y) where 'x' is np.array input and 'y' is np.array output. Used for testing ANN ;
                            example: [([1],[2])([3],[4])]
            ''evaluation_data'' - list of tuples (x,y) 'x' is np.array input and 'y' is np.array output. Used for validation of ANN (early stopping) ;
                                  example: [([1],[2])([3],[4])]
            ''dropout'' - list of True/False variables for each of hidden layers whetheter to use dropout of neurons on it ;
                          example: [True,False] meaning dropout of half of neuron in first hidden layer and not dropout of neurons in second hidden layer
            ''lmbda'' - variable of regularization rate (used in updating weights in gradient descent) ;
                        example: [0] meaning no regularization
            ''early_stop_epochs'' - number of epochs to early stop
                                    (if change in accuracy in last 'early_stop_epochs' of validation data < 'early_stop_variance' then delta will be updated) ;
                                    example: [10]
            ''early_stop_variance'' - minimum variance of last 'early_stop_epochs' of validation data
                                      (if variance less than 'early_stop_variance' then delta will be updated) ;
                                      example: [0.00001]
            ''early_stop_min_alpha'' - minimum alpha for early stop (if alpha was changed and its less or equal 'early_stop_min_alpha' then early stop)
                                       example: [alpha] meaning no updates in alpha, immediate early stop if variance is small
        """
        # early_stop_min_alpha is alpha/128 by default
        if (early_stop_min_alpha == None):
            early_stop_min_alpha = alpha / 128
        # dropout is false for each hidden layer by default
        if (dropout == None):
            dropout = [False for i in range(len(self.layer_dims) - 2)]
        if (len(dropout) != (len(self.layer_dims) - 2)):
            raise ValueError("dropout variable should be as number of hidden layers")

        # list for accuracies of validation_data, used in early stop to compute variance
        evaluation_corrects = []
        for j in range(epochs):

            if len(evaluation_corrects) >= early_stop_epochs:
                # if variance of last epochs is smaller than 'early_stop_variance'
                if (np.var(evaluation_corrects[-early_stop_epochs:]) < early_stop_variance):
                    # update alpha
                    print("Alpha Updated " + str(alpha) + " epoch:" + str(j))
                    alpha /= 2
                    # if alpha is too small
                    if alpha <= early_stop_min_alpha:
                        print("Early stopping at epoch: " + str(j))
                        # start testing
                        if (test_data != None):
                            print("Running Test Data:")
                            self.test(test_data)
                        return

            # error and correct for training data
            error = 0
            correct = 0

            # prepare mini_batches for training data
            random.shuffle(training_data)
            mini_batches = [training_data[i:i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                # get input and output
                x = np.vstack([i[0] for i in mini_batch])
                y = np.vstack([i[1] for i in mini_batch])

                # dropout masks for hidden layers
                dropout_masks = []
                for i in range(len(dropout)):
                    if dropout[i]:
                        # half of neurons will be deactivated
                        dropout_masks.append(np.random.randint(2, size=self.layer_dims[i + 1]))
                    else:
                        dropout_masks.append(np.ones(self.layer_dims[i + 1]))

                # forward propagation
                layers = [x] #input
                # hidden layers (including dropouts)
                for i in range(self.num_layers - 2):
                    if dropout[i]:
                        layers.append(self.activation_functions[i].compute(
                            (np.dot(layers[i], self.weights[i]) + self.biases[i]) * dropout_masks[i] * 2))
                    else:
                        layers.append(
                            self.activation_functions[i].compute(np.dot(layers[i], self.weights[i]) + self.biases[i]))
                # output layer
                if self.use_softmax:
                    layers.append(softmax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1]))
                else:
                    layers.append(np.dot(layers[-1], self.weights[-1]) + self.biases[-1])

                # error and accuracy calculation for training data
                error += self.cost_function.compute(layers[-1], y)
                for i in range(len(mini_batch)):
                    correct += int(np.argmax(layers[-1][i]) == np.argmax(y[i]))

                # backpropagation
                deltas = [self.cost_function.derivative(layers[-1], y)]
                for i in range(1, self.num_layers - 1):
                    deltas.append(
                        deltas[-1].dot(self.weights[-i].T) * self.activation_functions[-i].derivative(layers[-i - 1]) *
                        dropout_masks[-i])
                deltas.reverse()

                # updating weights (using L2) and biases
                for i in range(len(self.weights)):
                    self.weights[i] = (1 - alpha * (float(lmbda) / len(training_data))) * self.weights[i] - (
                            alpha / len(mini_batch)) * layers[i].T.dot(deltas[i])
                for i in range(len(self.biases)):
                    self.biases[i] -= alpha / len(mini_batch) * deltas[i].sum(axis=0, keepdims=True)

            # error and correct for validation data
            evaluation_correct = 0
            evaluation_error = 0
            if evaluation_data != None:
                x_evaluation = np.vstack([i[0] for i in evaluation_data])
                y_evaluation = np.vstack([i[1] for i in evaluation_data])

                # forward propagation
                layers = [x_evaluation]
                for k in range(self.num_layers - 2):
                    layers.append(self.activation_functions[k].compute(
                        np.dot(layers[k], self.weights[k]) + self.biases[k]))
                if self.use_softmax:
                    layers.append(softmax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1]))
                else:
                    layers.append(np.dot(layers[-1], self.weights[-1]) + self.biases[-1])

                # error and accuracy calculation for validation data
                evaluation_error += self.cost_function.compute(layers[-1], y_evaluation)
                for i in range(len(evaluation_data)):
                    evaluation_correct += int(np.argmax(layers[-1][i]) == np.argmax(y_evaluation[i]))
                # update last evaluation_corrects
                evaluation_corrects.append(evaluation_correct / len(evaluation_data))

            if test_data != None:
                # each 10 epochs or last epoch
                if j % 10 == 0 or j == epochs - 1:
                    # error and correct for test data
                    test_error = 0
                    test_correct = 0
                    x_test = np.vstack([i[0] for i in test_data])
                    y_test = np.vstack([i[1] for i in test_data])

                    # forward propagation
                    layers = [x_test]
                    for k in range(self.num_layers - 2):
                        layers.append(self.activation_functions[k].compute(
                            np.dot(layers[k], self.weights[k]) + self.biases[k]))
                    if self.use_softmax:
                        layers.append(softmax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1]))
                    else:
                        layers.append(np.dot(layers[-1], self.weights[-1]) + self.biases[-1])

                    # error and accuracy calculation for test data
                    test_error += self.cost_function.compute(layers[-1], y_test)
                    for i in range(len(test_data)):
                        test_correct += int(np.argmax(layers[-1][i]) == np.argmax(y_test[i]))

                    # prints of errors and accurcies
                    if evaluation_data:
                        print("\n" + "I:" + str(j) +
                              " Test-Err:" + str(test_error / float(len(test_data)))[0:5] +
                              " Test-Acc:" + str(test_correct / float(len(test_data))) +
                              " Train-Err:" + str(error / float(len(training_data)))[0:5] +
                              " Train-Acc:" + str(correct / float(len(training_data))) +
                              " Validation-Err:" + str(evaluation_error / float(len(evaluation_data)))[0:5] +
                              " Validation-Acc:" + str(evaluation_correct / float(len(evaluation_data))))
                    else:
                        print("\n" + "I:" + str(j) +
                              " Test-Err:" + str(test_error / float(len(test_data)))[0:5] +
                              " Test-Acc:" + str(test_correct / float(len(test_data))) +
                              " Train-Err:" + str(error / float(len(training_data)))[0:5] +
                              " Train-Acc:" + str(correct / float(len(training_data))))

    def test(self, test_data):
        """
        Network test function is responsible for testing given dataset.
        Parameters:
            ''test_data'' - list of tuples (x,y) where 'x' is np.array input and 'y' is np.array output. Used for testing ANN ;
                            example: [([1],[2])([3],[4])]

        """
        # error and correct for test data
        test_error = 0
        test_correct = 0
        x_test = np.vstack([i[0] for i in test_data])
        y_test = np.vstack([i[1] for i in test_data])

        # forward propagation
        layers = [x_test]
        for k in range(self.num_layers - 2):
            layers.append(self.activation_functions[k].compute(
                np.dot(layers[k], self.weights[k]) + self.biases[k]))
        if self.use_softmax:
            layers.append(softmax(np.dot(layers[-1], self.weights[-1]) + self.biases[-1]))
        else:
            layers.append(np.dot(layers[-1], self.weights[-1]) + self.biases[-1])

        # error and accuracy calculation for test data
        test_error += self.cost_function.compute(layers[-1], y_test)
        for i in range(len(test_data)):
            test_correct += int(np.argmax(layers[-1][i]) == np.argmax(y_test[i]))

        print(
            " Test-Err:" + str(test_error / float(len(test_data)))[0:5] +
            " Test-Acc:" + str(test_correct / float(len(test_data))))
