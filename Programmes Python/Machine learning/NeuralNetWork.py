#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:40:37 2020

@author: yanisfilippi
"""

import numpy as np
import matplotlib.pyplot as plt

#========================== Fonctions 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigPrime(x):
    return x*(1-x)

#========================== Classes 


#### Classe Brain composée des neurones 

class Brain:
    def __init__(self,nb_neurons_input,nb_neurons_hidden,nb_neurons_output,nb_hidden):
        self.__layers = [InputLayer(nb_neurons_input)]
        self.__layers.append(Layer(nb_neurons_hidden,nb_neurons_input))
        for x in range(nb_hidden-1):
            self.__layers.append(Layer(nb_neurons_hidden,nb_neurons_hidden))
        self.__layers.append(OutputLayer(nb_neurons_output,nb_neurons_hidden))
        
        self.__train_data = None
        self.__train_solution = None
        return
    
    def set_train_data(self,data):
        self.__train_data = data
        return
    
    def set_train_solution(self,solution):
        self.__train_solution = solution
        return
    
    def run_all(self,entry):
        '''
        Permet un passage dans le réseau neuronal. 
        (Un 'ALLER' avec comme entrée la variable entry {FEEDFORWARD})
        '''
        self.__layers[0].set_entry(entry)
        self.__layers[1].set_A_Nminus1(entry)
        for i in range(1,len(self.__layers)):
            self.__layers[i].run()
            if i+1!=len(self.__layers):
                self.__layers[i+1].set_A_Nminus1(self.__layers[i].get_A_N())
        return
    
    def correct_all(self,cost=False,coeff=1):
        '''
        Correction de tous les coefficients du réseau
        (passage 'RETOUR' et correction par calcul du gradient {BACKPROPAGATION})
        '''
        L_N = self.__layers[-1]
        L_Nminus1 = self.__layers[-2]
        L_N.set_exact(self.__train_solution)
        L_N.correct(coeff)
        L_Nminus1.set_W_N(L_N.get_W_Nminus1())
        L_Nminus1.set_Err_N(L_N.get_Err_Nminus1())
        for i in range(2,len(self.__layers)):
            L_N = self.__layers[-i]
            L_Nminus1 = self.__layers[-i-1]
            L_N.correct(coeff)
            L_Nminus1.set_W_N(L_N.get_W_Nminus1())
            L_Nminus1.set_Err_N(L_N.get_Err_Nminus1())
        if cost:
            return self.__layers[-1].cost()
        return
    
    def train(self,Data,Solution,Iter_Training,Cost_Rate=10,Coeff=1):
        '''
        Entrainement du réseau avec les variables Data et Solution
        On effectue un nombre d'iteration de l'entrainement : Iter_Training 
        
        Cost_Rate (default=10) le 'pas' du tracé de la courbe du cout en fonction du nb d'iterations
        '''
        self.set_train_data(Data)
        self.set_train_solution(Solution)
        Iter = []
        Cost = []
        for i in range(Iter_Training):
            self.run_all(self.__train_data)
            if Iter_Training%Cost_Rate==0:
                a = self.correct_all(True,Coeff)
                print('Training Iter N° :',i, 'Cost :',a)
                Iter.append(i)
                Cost.append(a)
            else:
                self.correct_all(coeff=Coeff)
        plt.plot(Iter,Cost)
        plt.show()
        return
    
    def test(self,entry):
        '''
        Méthode pour simplemennt faire traverser le réseau à la variable entry
        À utiliser une fois que le réseau est entrainé
        '''
        self.run_all(entry)
        out = self.__layers[-1].get_A_N()
        print('===========OutPut==========')
        print(out)
        return out

        
#Classe de couche, on est dans une couche (N) et les variables sont nommées d'après ce postulat
class Layer:
    def __init__(self,nb_neurons, nb_neurons_Nminus1):
        self.__nb_neurons = nb_neurons
        
        #N-1
        self.__A_Nminus1 = None
        self.__W_Nminus1 = 2*np.random.random((nb_neurons,nb_neurons_Nminus1)) - 1
        self.__B_Nminus1 = 2*np.random.random((nb_neurons,1)) - 1
        self.__Err_Nminus1 = None
        #N
        self.__A_N = np.zeros((nb_neurons,1))
        self.__W_N = None
        self.__Err_N = None
        
        return
    
    
    
    def get_A_N(self): 
        return self.__A_N
    
    def get_Err_Nminus1(self):
        return self.__Err_Nminus1
    
    def get_W_Nminus1(self):
        return self.__W_Nminus1
    
    def set_W_N(self,W):
        self.__W_N = W
        return
    def set_Err_N(self,E):
        self.__Err_N = E
        return
    
    def set_A_Nminus1(self,A):
        self.__A_Nminus1 = A
        return
    
    #Les méthodes précédents sont nécéssaire pour la communication synaptique entre couches
    
    def do_Err_Nminus1(self):
        self.__Err_Nminus1 = sigPrime(self.__A_N)*(np.dot(self.__W_N.T,self.__Err_N))
        return
    
    #la variable Err est une variable pivot pour eclaircir les calculs
    
    def run(self):
        '''
        Passage dans la couche (FEEDFORWARD)
        '''
        self.__A_N = sigmoid(np.dot(self.__W_Nminus1,self.__A_Nminus1) + self.__B_Nminus1)
        return
    
    def correct(self,coeff=1): 
        '''
        Correction des coefficiens de W_Nminus1 et B_Nminus1 
        '''
        if coeff!=1:
            print('ok')
        self.do_Err_Nminus1()
        (a,b) = np.shape(self.__Err_Nminus1)
        
        corr_W_Nminus1 = coeff*np.dot(self.__Err_Nminus1,self.__A_Nminus1.T)
        corr_B_Nminus1 = coeff*np.dot(self.__Err_Nminus1,np.ones((b,1)))
        
        sign = -1
        
        self.__W_Nminus1 += sign*corr_W_Nminus1
        self.__B_Nminus1 += sign*corr_B_Nminus1
        
        return


#On distingue les couches d'entrée et de sortie 

class InputLayer(Layer):
    def __init__(self,nb_neurons):
        self.__neurons = None
        self.__nb_neurons = nb_neurons
        return
    
    def set_entry(self,L):
        self.__neurons = L
        return 
    
    def run(self):
        print("Can't run entry")
        return
    
    def correct(self):
        print("Can't correct entry")
        return 

    
class OutputLayer:
    def __init__(self,nb_neurons, nb_neurons_Nminus1):
        self.__nb_neurons = nb_neurons
        
        #N-1
        self.__A_Nminus1 = None
        self.__W_Nminus1 = 2*np.random.random((nb_neurons,nb_neurons_Nminus1)) - 1
        self.__B_Nminus1 = 2*np.random.random((nb_neurons,1)) - 1
        self.__Err_Nminus1 = None
        #N
        self.__A_N = np.zeros((nb_neurons,1))
        self.__W_N = None
        self.__Err_N = None
        
        self.__exact = None
        return

    
    def set_exact(self,L):
        self.__exact = L
        return

    def cost(self):
        c = sum((self.__A_N - self.__exact)**2)
        p = len(c)
        return sum(c)/p
    
    def do_Err_Nminus1(self):
        self.__Err_N = 2*(self.__A_N - self.__exact)
        self.__Err_Nminus1 = sigPrime(self.__A_N)*self.__Err_N
        return

    def get_A_N(self): 
        return self.__A_N
    
    def get_Err_Nminus1(self):
        return self.__Err_Nminus1
    
    def get_W_Nminus1(self):
        return self.__W_Nminus1
    
    def set_W_N(self,W):
        self.__W_N = W
        return
    def set_Err_N(self,E):
        self.__Err_N = E
        return
    
    def set_A_Nminus1(self,A):
        self.__A_Nminus1 = A
        return
    
    def run(self):
        self.__A_N = sigmoid(np.dot(self.__W_Nminus1,self.__A_Nminus1) + self.__B_Nminus1)
        return
    
    def correct(self,coeff=1): 
        self.do_Err_Nminus1()
        (a,b) = np.shape(self.__Err_Nminus1)
        
        corr_W_Nminus1 = coeff*np.dot(self.__Err_Nminus1,self.__A_Nminus1.T)
        corr_B_Nminus1 = coeff*np.dot(self.__Err_Nminus1,np.ones((b,1)))
        
        sign = -1
        
        self.__W_Nminus1 += sign*corr_W_Nminus1
        self.__B_Nminus1 += sign*corr_B_Nminus1
        
        return
