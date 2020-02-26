#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:33:51 2020

@author: yanisfilippi
"""
import numpy as np
import NeuralNetWork as nnw
n = 5

#a = np.random.randint(n,size=(4,4))
#print(a)
#
#b = np.random.randint(n,size=(4,4))
#print(b)
#
##c = np.dot(a,b)
##print(c)
#
#d = a*b
#print(d)
#
##e = d + a
##print(e)
##
##print(len(b))
##
##f = a + b
##print(f)
#

inputs = np.array([ [0, 0, 0, 1],
                    [0.2, 0, 0.9, 1],
                    [0, 1, 1, 0.75],
                    [0.76, 0.22, 0.97, 0],
                    [1, 1, 1, 0.2]])
reponses = np.array([[0],
                     [0],
                     [1],
                     [0],
                     [1]])
print(reponses.T)
inputs_test = np.array([[1, 0.95, 1, 0],
                        [0.2, 1, 0.7, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]])
reponses_test = np.array([[1],
                         [1],
                         [0],
                         [0]])
c = np.ones(5)


Alexa = nnw.Brain(4,16,1,2)
Alexa.train(inputs.T,reponses.T,500)

Alexa.test(inputs_test.T)
print(reponses_test.T)