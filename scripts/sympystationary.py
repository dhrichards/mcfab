#%%
from sympy import *
from sympy.abc import l,m
import numpy as np
from tqdm import tqdm

# Define spherical coordinates
r = 1
theta, phi = symbols('theta,phi', real=True, positive=True)
psi = (theta, phi)
nt = (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta))
n = Matrix([nt[0], nt[1], nt[2]])


iota = symbols('iota', real=True, positive=True)
lamb = symbols('lambda', real=True, positive=True)





#du[2,2]=1-du[1,1]-du[0,0]
du = zeros(3)
du[0,0]=0.5
du[1,1]=0.5
du[2,2]=-1
D=0.5*(du+du.T)
W=0.5*(du-du.T)
w = 2*Matrix([W[2,1], W[2,0], W[1,0]])
    




def gradstar(A):
    g=((cos(theta)*cos(phi)*diff(A,theta) \
            - (sin(phi)/sin(theta))*diff(A,phi)),
           (cos(theta)*sin(phi)*diff(A,theta) \
            + (cos(phi)/sin(theta))*diff(A,phi)),
           (- sin(theta)*diff(A,theta)))
    
    return simplify(Matrix([g[0], g[1], g[2]]))

def divstar(v):
    g=((cos(theta)*cos(phi)*diff(v[0],theta) \
            - (sin(phi)/sin(theta))*diff(v[0],phi))+
           (cos(theta)*sin(phi)*diff(v[1],theta) \
            + (cos(phi)/sin(theta))*diff(v[1],phi))+
           (- sin(theta)*diff(v[2],theta)))
    
    return simplify(g)

iota = 1
#lamb = 0.5
Dnn = (D@n).dot(n)

vstar = W@n -iota*(D@n - Dnn*n)


# Define general function
f = Function('f')(theta)#,phi)

eqn = -divstar(vstar*f) +lamb*divstar(gradstar(f))



