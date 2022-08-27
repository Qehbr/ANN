import random
import numpy as np
import mnist_loader
from network import Network
from cross_entropy import CrossEntropy
from ReLU import ReLU


if __name__ == '__main__':

    # getting MNIST data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = training_data
    test_data = test_data

    # preparing MNIST data
    x_train = [i[0].T for i in training_data]
    y_train = [i[1].T for i in training_data]
    training_data = list(zip(x_train, y_train))

    x_test = [i[0].T for i in test_data]
    y_test = [i[1].T for i in test_data]
    one_hot_labels = np.zeros((len(y_test), 10))
    for i, l in enumerate(y_test):
        one_hot_labels[i][l] = 1
    y_test = one_hot_labels
    test_data = list(zip(x_test,y_test))

    x_validation = [i[0].T for i in validation_data]
    y_validation = [i[1].T for i in validation_data]
    one_hot_labels = np.zeros((len(y_validation), 10))
    for i, l in enumerate(y_validation):
        one_hot_labels[i][l] = 1
    y_validation = one_hot_labels
    validation_data = list(zip(x_validation, y_validation))

    # creating and fitting ANN
    ann = Network([784,50,10], CrossEntropy(), activation_functions=[ReLU()], use_softmax=True)
    ann.fit(training_data,alpha=0.15,epochs=200, mini_batch_size=100, test_data=test_data, evaluation_data=validation_data, dropout=[False], lmbda=0)



