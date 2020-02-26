from os import getcwd, chdir
import numpy as np 
import matplotlib.pyplot as plt
import NeuralNetWork as NNW


chdir('/Users/yanisfilippi/Desktop')



A = plt.imread('Cross&Full.png')

print(np.shape(A))




def BlackWhite(img):
    A = plt.imread(img)
    n, p, t = np.shape(A)
    for i in range(n):
        for j in range(p):
            m = (sum(A[i,j,0:3]))/3
            A[i,j,0:3] = np.array([m]*3) 
    return A



B = BlackWhite('Cross&Full.png')

Border = B[100,92,0]
print(Border)

def Split(B,Border):
    n,p,t = np.shape(B)
    H = []
    V = []
    
    for i in range(n):
        if B[i,10,0]==Border:
            H.append(i)
    for j in range(p):
        if B[10,j,0]==Border:
            V.append(j)
    def l(V):
        S = []
        tmpH = False
        t = []
        for i in range(1,len(V)):
            if V[i]-V[i-1]==1:
                if tmpH:
                    t.append(V[i])
                else:
                    t = [V[i-1],V[i]]
                    tmpH=True
            else:
                if tmpH:
                    S.append(t)
                t = []
                tmpH=False
        S.append(t)     
        return S
    LH = l(H)
    LV = l(V)
    L = [B[0:LH[0][0],:,:]]
    for i in range(len(LH)-1):
        L.append(B[LH[i][-1]+1:LH[i+1][0],:,:])
    L.append(B[LH[-1][-1]+1:-1,:,:])
    L2 = [[x[:,0:LV[0][0]]] for x in L]    
    for i in range(len(LV)-1):
        for p in range(len(L2)):
            L2[p].append(L[p][:,LV[i][-1]+1:LV[i+1][0],:])
    for p in range(len(L2)):
        L2[p].append(L[p][:,LV[-1][-1]+1:-1,:])
    return L2

def Square(B, x=None, ):
    n,p,t = np.shape(B)
    if x!=None:
        d1 = n-x
        d2 = p-x
        L = B[d1//2+1:-(d1//2+d1%2+1),d2//2+1:-(d2//2+d2%2+1),:]
    elif n==max(n,p):
        d = n-p 
        L = B[d//2:-(d//2+1),:,:]
    else:
        d = p-n
        L = B[:,d//2:-(d//2+1),:]
    return L

def FullSquare(S):
    n = len(S)
    p = len(S[0])
    P = [[] for x in range(n)]
    Min = []
    for i in range(n):
        for j in range(p):
            a, b, c = np.shape(S[i][j])
            d = min(a,b)
            Min.append(d)
#            print(d)
    D = min(Min)
    for i in range(n):
        for j in range(p):
            P[i] += [Square(S[i][j],D)]
#            print('j = ',j,' P{i} : ',len(P[i]))
            
    return P

def ShowAll(S):
    m = len(S)
    n = len(S[0])
    fig, AX = plt.subplots(m,n)
    for i in range(m):
        for j in range(n):
            AX[i,j].imshow(S[i][j])
            AX[i,j].axis('off')
    return

def Unwrap(B):
    n, p, t = np.shape(B)
    L = np.zeros(n*p)
    for i in range(n):
        L[i*n:(i+1)*n] = B[i,:,0]
    return L

S = Split(B,Border)
S = FullSquare(S)


data = np.zeros((86*86,14))
solution = np.array([[0,1,0,1,0,1,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,1,0,1,0,1,0,1]])
for y in range(2):
    for x in range(7):
        print(x,y)
        data[:,x+7*y] = Unwrap(S[y][x])

print(np.shape(data))
print(np.shape(solution))

Alexa = NNW.Brain(7396,10,2,2)
Alexa.train(data,solution,2000)

