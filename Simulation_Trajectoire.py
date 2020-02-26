#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:14:21 2020

@author: yanisfilippi
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


######################################################
#===========                             ============#
#===========  Simulation de Trajectoire  ============#
#===========                             ============#
######################################################


# Pas et nombre de points simulés 
h = 10**(-3)
n = 3000


U0 = np.array([[0,0,0],[0,0,0.2]]) #(vit, pos)
z0 = U0[1,2]

W = 200 # En tr/s
theta = -135 # En ° (angle y,z : 0° -> w=y, 90° -> w=z)
phi = 0 # En ° (angle z,x : 0° -> w=z, 90° -> w=x)
w = 2*np.pi*W*np.array([np.cos(theta*np.pi/180),np.sin(phi*np.pi/180),np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)])

#Initialisation des listes de positions

X = [U0[1,0]]
Y = [U0[1,1]]
Z = [U0[1,2]]

print('Simulation :',n*h,'s')

#=============== Paramètres de la simulation =============#

alpha = 1.2*(10**(-3))  #Coeff de frottement visqueux 
m = 2.7*(10**(-3))      #Masse Balle
g = 9.81                #Champ gravitationnel
magnus = 1.8*(10**(-4)) #Coeff Magnus 
µ = 0.2526              #Frottement au rebond
r = 2*10**(-2)          #Rayon balle

#=============== Fonctions Utiles ===================#

def vect(u,v):  #Simplifie les expressions (produit vectoriel)
    w = np.zeros(3)
    w[0] = u[1]*v[2]-u[2]*v[1]
    w[1] = u[2]*v[0]-u[0]*v[2]
    w[2] = u[0]*v[1]-u[1]*v[0]
    return w

def f(U,w):     #Fonction pour la méthode d'Euler
    du = U[0,:]
    d2u = -alpha*(np.sqrt(sum(du*du)))*du/m - g*np.array([0,0,1]) + magnus*vect(w,du)
    return np.array([d2u,du])

def norm(a):    #Simplifie également (norme 2)
    n = np.sqrt(sum(a**2))
    return n

def e(vz):      #Coeff de restitution en fonction de Vz
    Vz = abs(vz)
    if Vz<1.9:
        return 0.93
    else:
        return 1 - 0.037*Vz

#=============== Gestion du rebond ====================#

def rebond(U,r,w):  #Fonction qui permet de gérer le rebond
    Vc = U[0,:] + vect(np.array([0,0,r]),w)
    E = e(U[0,2])
    beta = 2.5*µ*(1+E)*abs(U[0,2])/norm(Vc)
    if beta>=1: ###Cas de Roulement
        tau = 0.4       #tau correspond au symbole alpha dans la théorie
        c = (1-tau)/r   # (ou le symbole proportionnel)
        d = tau
    else:       ###Cas de Glissement 
        tau = beta/2.5
        c = 3*tau/(2*r)     
        d = 1 - 3*tau/2
    THETA = np.zeros((6,6))
    A = np.diagflat([1-tau,1-tau,-E])
    B = np.diagflat([tau*r,0],1) + np.diagflat([-tau*r,0],-1)
    C = np.diagflat([-c,0],1) + np.diagflat([c,0],-1)
    D = np.diagflat([d,d,1])
    THETA[0:3,0:3] = A
    THETA[0:3,3:6] = B #A et B sont les mêmes quelque soit beta
    THETA[3:6,0:3] = C #C et D changent d'ou les variables c et d
    THETA[3:6,3:6] = D
    VW = np.zeros(6)
    VW[0:3] = U[0,:] 
    VW[3:6] = w
    VW2 = np.dot(THETA,VW) #Multiplication matricielle
    return VW2 


#================== Simulation =====================#

for p in range(n): #Application de la méthode d'Euler
    U0 = U0 + h*f(U0,w)
    z0 = U0[1,2]
    X.append(U0[1,0])
    Y.append(U0[1,1])
    Z.append(U0[1,2]) #Cas d'un rebond : 
    if z0<10**(-3) and sum(U0[0]*np.array([0,0,1]))<=0: 
        VW = rebond(U0,r,w)
        U0[0,:] = VW[0:3]
        w = VW[3:6]


#================== Tracé en 3D =====================#

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot(X,Y,Z)
ax.set_xlabel('X')
ax.set_xlim3d([-1,1])
ax.set_ylabel('Y')
ax.set_ylim3d([0,2])
ax.set_zlabel('Z')
ax.set_zlim3d([-0.2,1])
fig.show()

#===================================================#
#===================================================#
#===================================================#











