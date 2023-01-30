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


def numpy2sympy(A):
    return Matrix(A)

def sympy2numpy(A):
    return np.array(A.tolist()).astype(np.float64)

class gbmexact:
    def __init__(self):
        self.theta, self.phi = symbols('theta,phi', real=True, positive=True)
        self.psi = (theta, phi)
        self.nt = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        self.n = Matrix([nt[0], nt[1], nt[2]])
        self.f  = 1/(4*pi)


    def A2calc(self,f):
        # Calculate 2nd order orientation tensor
        A = zeros(3,3)
        for i in range(3):
            for j in range(3):
                A[i,j] = self.integrate_sphere(f*self.nt[i]*self.nt[j])

        return A



    def integrate_sphere(self,f):
        return integrate(f*sin(self.theta),(self.theta,0,pi),(self.phi,0,2*pi))

    
    def solve_constant(self,gradu,dt,tmax,beta):
        self.dt = dt
        self.tmax = tmax
        self.t  = np.arange(0,tmax,dt)
        self.nsteps = len(self.t)

        self.gradu = numpy2sympy(gradu)
        D = 0.5*(self.gradu+self.gradu.T)
        W = 0.5*(self.gradu-self.gradu.T)

        self.a2 = np.zeros((self.nsteps,3,3))


        self.beta = beta

        self.Defstar = 5*((D*n).dot(D*n) - (n.dot(D*n))**2)/(trace(D*D))

        for i in tqdm(range(self.nsteps)):
            A = self.A2calc(self.f)
            self.a2[i,:,:] = sympy2numpy(A)
            rhs = self.f*self.beta*(self.Defstar - self.integrate_sphere(self.Defstar*self.f))
            self.f = rhs*self.dt + self.f





# %%
