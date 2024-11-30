##########################################################
#           PERCEPTRON ALGORITHM FROM SCRATCH            #
##########################################################

# import packages
import numpy as np
import pandas as pd

# class for Perceptron algorithm
class PerceptronAlgorithm(object):
    
    # hyperparameters definition
    def __init__(self, eta, max_epochs, threshold):
        self.eta = eta
        self.max_epochs = max_epochs
        self.threshold = threshold
    
    # random initialization of weights and biases
    def get_weights(self, n):
        self.w = np.random.rand(n)
        self.b = np.random.rand(1)
    
    # linear combination
    def input_net(self, x):
        net = np.dot(x, self.w) + self.b
        return net
    
    # activation function heaviside
    def f(self, net):
        return 1 if net >= 0 else -1
    
    # make prediction results
    def predict(self, x):
        y_pred = self.f(self.input_net(x))
        return y_pred
    
    # loss function
    def loss_fn(self, y, y_pred):
        loss = (y - y_pred)
        return loss        
    
    # training step
    def fit(self, x_train, y_train):
        x_train = np.array(x_train, dtype=float)
        y_train = np.array(y_train, dtype=int)
        n = x_train.shape[0]
        self.get_weights(x_train.shape[1])
        cost = []
        count = 0
        E = 2 * self.threshold
        while E >= self.threshold and count < self.max_epochs:
            E = 0.0
            for i in range(n):
                xi = x_train[i]
                yi = y_train[i]
                y_hat = self.predict(xi)
                error = self.loss_fn(yi, y_hat)
                E += error ** 2
                self.w += self.eta * error * xi
                self.b += self.eta * error
            count += 1
            E /= (2 * n)
            cost.append(E)
            print(f'Epoch {count}: Error = {E}')
        self.loss = E
        self.cost_ = cost
    
    # function to make iterative process of test
    def test(self, x_test, y_test):
        n = x_test.shape[0]
        self.accuracy = 0 
        y_pred = list()
        
        for i in range(n):
            xi = x_test[i, :]
            yi = y_test[i]
            y_pred.append(self.predict(xi))

            # verify correct classification            
            if y_pred[i] == yi:
                self.accuracy = self.accuracy + 1
        
        # calculate accuracy
        self.accuracy = 100 * round(self.accuracy/n, 5)
        
        return y_pred
