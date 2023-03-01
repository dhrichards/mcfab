#%%
# Script to compare shtns and original implementation of speccaf
import numpy as np
import matplotlib.pyplot as plt
from speccaf.speccaf_shtns import solver as solver_shtns
import speccaf.spherical as spherical
import speccaf.solver as solver
from meso_fab_mc import meso_mc as mc
from tqdm import tqdm
import exactsympy as es
import closuresvect as closure


lmax = 12
mmax = 6

# np.random.seed(0)
# gradu = np.random.rand(3,3)
# gradu[2,2] = -gradu[0,0] - gradu[1,1]

gradu = np.zeros((3,3))
gradu[0,0] = 0.5
gradu[1,1] = 0.5
gradu[2,2] = -1
# gradu[0,2] = 1
#gradu[0,1] = 2

# gradu = np.array([[ 2.62396833e-04,  4.28081245e-04,  0.00000000e+00],
#        [ 1.57583637e-04, -1.80731124e-04,  0.00000000e+00],
#        [ 0.00000000e+00,  0.00000000e+00, -8.16657096e-05]])



dt = 2
tmax = 10
T = -5
x = [1,0.5,0]
#x=None
## Discrete

disc = mc.solver(5000)
disc.solve_constant(gradu,dt,tmax,x)


##A2 evolution
def da2(a2,gradu,x):
    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)

    D2 = np.einsum('ij,ji',D,D)
    effectiveSR = np.sqrt(0.5*D2)

    iota = x[0]
    lamb = x[1]*effectiveSR




    a4 = closure.compute_closure(a2)
        

    da2 = np.einsum('ik,kj->ij',W,a2) - np.einsum('ik,kj->ij',a2,W)\
          -iota*(np.einsum('ik,kj->ij',D,a2) + np.einsum('ik,kj->ij',a2,D)\
                 - 2*np.einsum('ijkl,kl->ij',a4,D)) + lamb*(np.eye(3) - 3*a2)
    
    return da2


def a2evolution(gradu,dt,tmax,x=None):
    nt = disc.nsteps
    a2 = np.zeros((nt,3,3))

    a2[0,...] = np.eye(3)/3

    for i in range(nt-1):
        a2[i+1,...] = a2[i,...] + dt*da2(a2[i,...],gradu,x)

    return a2


A = a2evolution(gradu,dt,tmax,x)



# nsteps = int(tmax/dt)
# disc.a2 = np.zeros((nsteps,3,3))
# for i in tqdm(range(nsteps)):
#     disc.a2[i,...] = mc.a2calc(disc.n,disc.m)
#     disc.iterate(gradu,dt,x)
# disc.t = np.arange(nsteps)*dt

## Shtns implementation

sc = solver_shtns(12,mmax)

sc.solve_constant(gradu,T,dt,tmax,x=x)
a2_sc = sc.a2(sc.f)

#eigvals_sht = np.linalg.eigvals(a2_sc)


nt = sc.t.size

## Original implementation

sh = spherical.spherical(lmax)


# Calculate fabric tensor
a2=np.zeros((nt,3,3),dtype='complex128')


f0 = sh.fabricfromdiagonala2(1/3)

f=np.zeros((nt,f0.size),dtype='complex128')
f[0,:]=f0
a2[0,...] = sh.a2(f[0,:])

for i in range(nt-1):


    #Update fabric with dt T[i] gradu[i]
    rk = solver.rk3iterate(T, gradu, sh,x=[x[0],x[1]/2,x[2]])
    f[i+1,:] = rk.iterate(f[i,:], dt)

    # Update orientation tensors
    a2[i+1,...] = sh.a2(f[i+1,:])

# eigvals_old = np.linalg.eigvals(a2)


# plt.plot(sc.t,a2_sc[:,0,0],'r')
# plt.plot(sc.t,a2_sc[:,1,1],'g')
# plt.plot(sc.t,a2_sc[:,2,2],'b')
# plt.plot(sc.t,a2_sc[:,0,2],'k')

plt.plot(disc.t,A[:,0,0],'r',label='a2 evolution')
plt.plot(disc.t,A[:,1,1],'g')
plt.plot(disc.t,A[:,2,2],'b')
#plt.plot(disc.t,A[:,0,2],'k')

plt.plot(sc.t,a2[:,0,0],'r--',linewidth=2,label='SpecCAF')
plt.plot(sc.t,a2[:,1,1],'g--',linewidth=2)
plt.plot(sc.t,a2[:,2,2],'b--',linewidth=2)
#plt.plot(sc.t,a2[:,0,2],'k--',linewidth=2)

plt.plot(disc.t,disc.a2[:,0,0],'r:',label='Discrete')
plt.plot(disc.t,disc.a2[:,1,1],'g:')
plt.plot(disc.t,disc.a2[:,2,2],'b:')
#plt.plot(disc.t,disc.a2[:,0,2],'k:')

plt.legend()

plt.ylim(-0.1,1)


