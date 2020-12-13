#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 9 13:56:42 2020

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
    
    
    def batcher(self, X, Y, bs):
        p = X.shape[1]
        li, ui = 0, bs
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li + bs, ui + bs
            else:
                return None
                
    
    # Entrenamiento.
    def train(self, X, Y, epochs=50, batch_size=20):
        p = X.shape[1] # Obtener la cantidad de patrones.
        
        for _ in range(epochs):
            minibatch = self.batcher(X, Y, batch_size)
            
            for mX, mY in minibatch:
                y_est = self.predict(mX)
                self.w += (self.eta / p) * np.dot((mY - y_est), mX.T).ravel()
                self.b += (self.eta / p) * np.sum(mY - y_est)
                  
# mBGD            
# Ejemplo.
p = 200
x = -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 + 3.5 * np.random.randn(p)
plt.plot(x, y, ".b")

neuron = LinearNeuron(1, 0.1)
neuron.train(x, y, epochs=200, batch_size=1)

# Dibujar linea.
xn = np.array([[-1, 1]])
plt.plot(xn.ravel(), neuron.predict(xn), "--k")
plt.savefig("LN.eps", format="eps")