#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 6 11:46:41 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt

class  LinearNeuron:
    def __init__(self, n_inputs, learning_rate = 0.1):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate # Factor de aprendizaje.


    # Predicción.
    def predict(self, X):
        Y_est = np.dot(self.w, X) + self.b # Propagación.
        return Y_est
    
    
    # Entrenamiento.
    def train(self, X, Y, epochs = 50, solver ="SGD"):
        p = X.shape[1] # Obtener la cantidad de patrones.
        if solver == "SGD":
            for _ in range(epochs):
                for i in range(p):
                    y_est = self.predict(X[:, i])
                    self.w += self.eta * (Y[:, i] - y_est) * X[:, i]
                    self.b += self.eta * (Y[:, i] - y_est)
                    
        elif solver == "BDG":
              for _ in range(epochs):
                    y_est = np.dot(self.w, X) + self.b
                    self.w += (self.eta / p) * np.dot((Y - y_est), X.T).ravel()
                    self.b += (self.eta / p) * np.sum(Y - y_est)
                  
        else: # Direct
            X_hat = np.concatenate((np.ones((1,p)), X), axis = 0)
            w_hat = np.dot(Y.reshape(1, -1), np.linalg.pinv(X_hat))
            self.b = w_hat[0, 0]
            self.w = w_hat[0, 1:]
            
            
# Ejemplo.
p=500 # p = 200
x = -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 + 30 * np.random.rand(p)
plt.plot(x, y, '.b')

neuron = LinearNeuron(1, 0.1)
neuron.train(x, y, epochs=200, solver="Direct")

# Dibujar linea.
xn = np.array([[-1, 1]])
plt.plot(xn.ravel(), neuron.predict(xn), "--k")
plt.savefig("Direct.eps", format="eps")