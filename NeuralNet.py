import numpy as np


class NeuralNet():
    def __init__(self, X:np.array, y:np.array, W:np.array, learningRate:float, layers:np.array, activationFunction: str):
        self.X = X
        self.y = y
        self.W = W
        self.output = y.shape()
        self.lr = learningRate
        self.layers = layers
        self.activationFunction = activationFunction
        self.epochs

    def neuron(self,activation:function):
        z = activation(np.dot(self.X,self.W))
        return z

    def relu(self, z):
        if z > 0:
            return z
        else:
            return 0

    def sigmoid(self, z):
        return 1/(1 + np.exp ^ -z)

    def tanh(self, z):
        return (np.exp ^ z - np.exp ^ -z) / (np.exp ^ z + np.exp ^ -z)

    def binaryStep(self, z):
        if z < 0:
            return 0
        else:
            return 1

    def softPlus(self, z):
        return np.log(1 + np.exp ^ z) / np.log(np.exp)

    def sgd(self, y, y_pred):
        for j in self.W:
            self.W[j] = self.W[j] - self.lr*(self.loss(y,y_pred))
        return self.W

    def loss(self, y, y_pred):
        # Cross entropy
        n = len(self.X)
        crossEntropy = [] 
        for i in n:
            loss =  -(1/n) * ((y[i]* np.log(y_pred[i])) + (1-y[i]) * np.log(1- y_pred[i]))
            crossEntropy.append(loss)
        return crossEntropy

    def train(self):
        #train neural net
        return