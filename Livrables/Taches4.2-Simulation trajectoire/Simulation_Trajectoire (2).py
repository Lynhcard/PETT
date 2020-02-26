#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:14:21 2020

@author: yanisfilippi
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



h = 10**(-3)
n = 3000

U0 = np.array([[0,4,-1],[0,0,0.2]])
z0 = U0[1,2]

W = 200 # En tr/s
theta = 90 # En °
w = 2*np.pi*W*np.array([np.cos(theta*np.pi/180),0,np.sin(theta*np.pi/180)])


X = [U0[1,0]]
Y = [U0[1,1]]
Z = [U0[1,2]]

print('Simulation :',n*h,'s')

def vect(u,v):
    w = np.zeros(3)
    w[0] = u[1]*v[2]-u[2]*v[1]
    w[1] = u[2]*v[0]-u[0]*v[2]
    w[2] = u[0]*v[1]-u[1]*v[0]
    return w

alpha = 1.2*(10**(-3))
m = 2.7*(10**(-3))
g = 9.81
magnus = 1.8*(10**(-5))

def f(U):
    du = U[0,:]
    d2u = -alpha*(np.sqrt(sum(du*du)))*du/m - g*np.array([0,0,1]) + magnus*vect(w,du)
    return np.array([d2u,du])

for p in range(n):
    U0 = U0 + h*f(U0)
    z0 = U0[1,2]
    X.append(U0[1,0])
    Y.append(U0[1,1])
    Z.append(U0[1,2])
    if z0<10**(-3) and sum(U0[0]*np.array([0,0,1]))<=0:
        U0[0,2] = -U0[0,2]
        U0[0,:] = np.sqrt(0.9)*U0[0,:]




#U0[0,2] = -U0[0,2]
#U0[0,:] = np.sqrt(0.9)*U0[0,:]
#
#
#for q in range(400):
#    U0 = U0 + h*f(U0)
#    z0 = U0[1,2]
#
#    #List_U[i+1,:,:] = U0
#    #List_t[i] = (i+1)*h
#    X.append(U0[1,0])
#    Y.append(U0[1,1])
#    Z.append(U0[1,2])
#    i+=1
#
#plt.plot(Y,Z)
#plt.axis('equal')
#plt.grid()
#plt.show()

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