#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:25:30 2019

@author: yanisfilippi
"""

#Entrée P = (X,Y,Z) tuple de 3 array
#Échantillonnage Fs = 100Hz 
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

#================ Extraction des données ==================

data = pd.read_csv('/Users/yanisfilippi/Desktop/1A/PE_TT/ComeXCyp.txt')
dt = data.values    
Index = data.keys()
n = len(Index)
DataList = []
Pts = []
Ts = 1/100
for i in range(n):
    if Index[i][:7] != 'Unnamed':
        A = dt[:,i:i+3]
        DataList.append((Index[i],A))
        Pts.append(Index[i])

n = len(DataList)
print(Pts)


#================ Calculs de paramètres ==================

def get(i,show=False):
    name, A = DataList[i]
    
    A = A[1:705,:]
    X = [float(s) for s in A[:,0]]
    Y = [float(s) for s in A[:,1]]
    Z = [float(s) for s in A[:,2]]
    if show:
        print(name)
    return np.array([X,Y,Z])

def vit(i):
    x,y,z = get(i)
    n = len(x)
    V = []
    for u in [x,y,z]:
        a = 100*(u[1:n]-u[0:n-1])
        V.append(a)
    return np.array(V)

def acc(i):
    Vx,Vy,Vz = vit(i)
    n = len(Vx)
    A = []
    for u in [Vx,Vy,Vz]:
        a = 100*(u[1:n]-u[0:n-1])
        A.append(a)
    return np.array(A)


#A = np.linspace(0,5,500)
#X, Y, Z = (25 - A**2)/25, 0*A, 0*A
#Vx = 100*(X[1:500]-X[0:499])
#Ax = 100*(Vx[1:499]-Vx[0:498])
#plt.clf()
#plt.plot(A,X)
#plt.plot(A[0:499],Vx)
#plt.plot()
#plt.show()



#=============== Courbes ===================


iBall = 2 #Index de la balle dans la liste des points 

ListAppearence = list(range(10,700))  #liste des frames pour lesquels l'enregistrement est utilisable

def PlotTop():
    Ax = [get(iBall)[0,i] for i in ListAppearence]
    Ay = [get(iBall)[1,i] for i in ListAppearence]
    p = ListAppearence[0]
    Bx = [get(i)[0,p] for i in [0,1,2,3,0]]
    By = [get(i)[1,p] for i in [0,1,2,3,0]]
    plt.plot(Ax,Ay)
    plt.plot(Bx,By)
    plt.axis('equal')
    return



def rebond():
    '''
    Détection des rebonds sur la table, renvoi une liste des frames auquels occurent ces rebonds
    '''
    P = get(iBall)[2]
    V = vit(iBall)[2]
    n = len(V)
    r = []
    for i in range(1,n):
        if ListAppearence[1:].count(i)>0:
            if V[i]>=0 and V[i-1]<0:
                if P[i]<50:
                    r.append(i)
    return r

def anglesRebond(n,f=100,p=10,decimales=2):
    '''
    Renvoi l'angle d'incidence de la balle sur la table (calculée avec le vecteur vitesse)
    Trace également les p frames avant et apres le rebond, avec vecteur moyen.
    
    n : numéro du rebond dans la liste des rebonds détectés
    f : facteur d'échelle sur le vecteur moyen de vitesse
    p : nombre de frame sur lesquels la moyenne est calculée
    decimales : nombre de décimales voulues pour le calcul de l'angle
    '''
    R = rebond()[n]
    P_av = get(iBall)[:,R-p:R]
    P_ap = get(iBall)[:,R:R+p]
    V_av = vit(iBall)[:,R-p:R]
    V_ap = vit(iBall)[:,R:R+p]
    fig= plt.figure()
    ax = p3.Axes3D(fig)
    Xm = np.array([sum(P_av[0])/p,sum(P_ap[0])/p])
    Ym = np.array([sum(P_av[1])/p,sum(P_ap[1])/p])
    Zm = np.array([sum(P_av[2])/p,sum(P_ap[2])/p])
    Um = np.array([sum(V_av[0])/p,sum(V_ap[0])/p])
    Vm = np.array([sum(V_av[1])/p,sum(V_ap[1])/p])
    Wm = np.array([sum(V_av[2])/p,sum(V_ap[2])/p])
    
    X = np.array(list(P_av[0])+list(P_ap[0]))
    Y = np.array(list(P_av[1])+list(P_ap[1]))
    Z = np.array(list(P_av[2])+list(P_ap[2]))
    U = np.array(list(V_av[0])+list(V_ap[0]))
    V = np.array(list(V_av[1])+list(V_ap[1]))
    W = np.array(list(V_av[2])+list(V_ap[2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    Norm = max(np.sqrt(U**2+V**2+W**2))
    Ang_Av = round(angle([Vm[0],Um[0],Wm[0]],[Vm[0],Um[0],0]),decimales)
    Ang_Ap = round(angle([Vm[1],Um[1],Wm[1]],[Vm[1],Um[1],0]),decimales)
    print(Ang_Av,Ang_Ap)
    ax.quiver(Xm,Ym,Zm,f*Um/Norm,f*Vm/Norm,f*Wm/Norm)
    ax.plot(X,Y,Z,'r:')
    ax.set(title='Avant : '+str(Ang_Av)+'° Après : '+str(Ang_Ap)+'°')
    fig.show()
    return

def angle(u,v):
    '''
    Calcul de l'angle entre deux vecteurs 
    '''
    U2 = [x**2 for x in u]
    V2 = [x**2 for x in v]
    NormU = np.sqrt(sum(U2))
    NormV = np.sqrt(sum(V2))
    S = [x*y for x,y in zip(u,v)]
    Scal = sum(S)
    return 180*np.arccos(Scal/(NormU*NormV))/np.pi

def fRebond(n,t='Accélération',p=10):
    '''
    Calcul de la <grandeur t> moyenne avant et après le rebond n, calcul sur p frames
    '''
    R = rebond()[n]
    if t=='Accélération':
        f = acc
        print('acc')
    if t=='vitesse':
        f = vit
        print('vit')
    A_av = f(iBall)[:,R-p:R]
    A_ap = f(iBall)[:,R:R+p]
    A_avX = sum(A_av[0])/p
    A_avY = sum(A_av[1])/p
    A_avZ = sum(A_av[2])/p
    A_apX = sum(A_ap[0])/p
    A_apY = sum(A_ap[1])/p
    A_apZ = sum(A_ap[2])/p
    NormAv = np.sqrt(A_avX**2+A_avY**2+A_avZ**2)
    NormAp = np.sqrt(A_apX**2+A_apY**2+A_apZ**2)
    
    return NormAv, NormAp 


    
def plotXYZ():
    '''
    Plot de X(t), Y(t) et Z(t) 
    '''
    A = get(iBall)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    X = [A[0,i] for i in ListAppearence]
    Y = [A[1,i] for i in ListAppearence]
    Z = [A[2,i] for i in ListAppearence]
    T = [i*Ts for i in range(len(ListAppearence))]
    ax1.plot(T,X)
    ax2.plot(T,Y)
    ax3.plot(T,Z)
    ax1.set_ylabel('X')
    ax2.set_ylabel('Y')
    ax3.set_ylabel('Z')
    plt.show()
    return

def plotU(n):
    '''
    Plot de la position/vitesse/accélération en fonction du temps selon la n-ième direction
    Si on choisit la direction Z ; affichage des rebonds détectés 
    '''
    Pos = get(iBall)[n]
    Vit = vit(iBall)[n]
    Acc = acc(iBall)[n]
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    Pos = [Pos[i] for i in ListAppearence]
    Vit = [Vit[i] for i in ListAppearence]
    Acc = [Acc[i] for i in ListAppearence]
    T = [i*Ts for i in range(len(ListAppearence))]
    if n==2:
        R = rebond()
        for i in R:
            if ListAppearence.count(i)>0:
                I = i-ListAppearence[0]
                print('Pos :',Pos[I-1],'==',Pos[I])
                ax1.plot([T[I]],[Pos[I]],'k.')
    ax1.plot(T,Pos)
    ax1.plot([T[0],T[-1]],[0,0],'r:')
    ax2.plot(T,Vit)
    ax2.plot([T[0],T[-1]],[0,0],'r:')
    ax3.plot(T,Acc)
    ax3.plot([T[0],T[-1]],[0,0],'r:')
    ax1.set_ylabel('Position')
    ax2.set_ylabel('Vitesse')
    ax3.set_ylabel('Accélération')
    if n==0:
        ax1.set_title('Direction X')
    elif n==1:
        ax1.set_title('Direction Y')
    else:
        ax1.set_title('Direction Z')
    plt.show()
    return


#=============== Animation Du Skelette ===================

def Skeleton(p, Skel_Sequence=None):
    '''
    Crée le squelette à la frame p
    '''
    SkelX = []
    SkelY = []
    SkelZ = []
    L = Skel_Sequence
    for i in L:
        SkelX.append(get(i)[0,p])
        SkelY.append(get(i)[1,p])
        SkelZ.append(get(i)[2,p])
    Skel = [SkelX,SkelY,SkelZ]
    return Skel  

def UPDATE(num):
    '''
    Update le squelette pour l'animation
    '''
    A = Skeleton(I_init+num,Seq)
    B = get(iBall)[:,I_init:I_init+num+1]
    AX.set_data(A[0:2])
    AX.set_3d_properties(A[2])
    AX2.set_data(A[0:2])
    AX2.set_3d_properties(A[2])
    AX3.set_data(get(iBall)[:2,num+I_init])
    AX3.set_3d_properties(get(iBall)[2,num+I_init])
    if Traj:
        AX4.set_data(B[0],B[1])
        AX4.set_3d_properties(B[2])
    return 


Traj = True
vitesse = 3 #Entier
I_init = 10
long = 700
Seq = [9,8,7,6,10,5,7,4,10,11,12,13,15,14,12,14,15,16,17,18,15,17,18,16]  #Séquence décrivant le squelette
ANIM = Skeleton(I_init,Seq)
fig1 = plt.figure()
ax = p3.Axes3D(fig1)
AX = ax.plot(ANIM[0],ANIM[1],ANIM[2],'r-')[0]
AX2 = ax.plot(ANIM[0],ANIM[1],ANIM[2],'bo')[0]
Ball = [[x] for x in get(iBall)[:,I_init]]
TBall = get(iBall)[:,I_init:I_init+1]
AX3 = ax.plot(*Ball,'ko')[0]
if Traj:
    AX4 = ax.plot(*TBall,'k:')[0]
ax.set_xlabel('X')
ax.set_xlim3d([-550,950])
ax.set_ylabel('Y')
ax.set_ylim3d([-500,2000])
ax.set_zlabel('Z')
ax.set_zlim3d([-250,1250])

Anim = animation.FuncAnimation(fig1, UPDATE, range(0,long,vitesse),interval=1)
plt.show()

#Modélisation de la table : pas très utile.
#x = [sum(get(i)[0])/704 for i in range(4)]     
#y = [sum(get(i)[1])/704 for i in range(4)]
#z = [-50 for i in range(4)]
#verts = [list(zip(x,y,z))]
#ax.add_collection3d(Poly3DCollection(verts))






