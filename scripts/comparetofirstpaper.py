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


gradu = np.zeros((3,3))
gradu[0,2]=1
dt = 0.01
tmax = 3
T = -20
#x = [1,0.5,0]
x=None
## Discrete

disc = mc.solver(5000)
disc.solve_constant(gradu,dt,tmax,T)

nt = disc.nsteps
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
    rk = solver.rk3iterate(T, gradu, sh)
    f[i+1,:] = rk.iterate(f[i,:], dt)

    # Update orientation tensors
    a2[i+1,...] = sh.a2(f[i+1,:])

eigvals_sc = np.linalg.eigvals(a2.real)
eigvals_disc = np.linalg.eigvals(disc.a2)

# Sort eigenvalues
eigvals_sc = np.sort(eigvals_sc,axis=1)
eigvals_disc = np.sort(eigvals_disc,axis=1)

plt.plot(disc.t,eigvals_disc[:,0],'r:',label='Discrete')
plt.plot(disc.t,eigvals_disc[:,1],'g:')
plt.plot(disc.t,eigvals_disc[:,2],'b:')

plt.plot(disc.t,eigvals_sc[:,0],'r--',label='SpecCAF')
plt.plot(disc.t,eigvals_sc[:,1],'g--')
plt.plot(disc.t,eigvals_sc[:,2],'b--')



plt.legend()

plt.ylim(-0.1,1)
plt.grid()


