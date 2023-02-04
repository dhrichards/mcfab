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


lmax = 6
mmax = 6

# np.random.seed(0)
# gradu = np.random.rand(3,3)
# gradu[2,2] = -gradu[0,0] - gradu[1,1]

gradu = np.zeros((3,3))
gradu[0,0] = -0.5
gradu[1,1] = -0.5
gradu[2,2] = +1
#gradu[0,2] = 1
# gradu[0,1] = 1

# gradu = np.array([[ 2.62396833e-04,  4.28081245e-04,  0.00000000e+00],
#        [ 1.57583637e-04, -1.80731124e-04,  0.00000000e+00],
#        [ 0.00000000e+00,  0.00000000e+00, -8.16657096e-05]])

T = -10

dt = 0.1
tmax = 5
x = [1.0 , 0.1, 0.0]

## Discrete

disc = mc.solver(5000,1e-2)
disc.solve_constant(gradu,dt,tmax,x,method='Static')



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
    rk = solver.rk3iterate(T, gradu, sh,x=x)
    f[i+1,:] = rk.iterate(f[i,:], dt)

    # Update orientation tensors
    a2[i+1,...] = sh.a2(f[i+1,:])

# eigvals_old = np.linalg.eigvals(a2)


plt.plot(sc.t,a2_sc[:,0,0],'r')
plt.plot(sc.t,a2_sc[:,1,1],'g')
plt.plot(sc.t,a2_sc[:,2,2],'b')
#plt.plot(sc.t,a2_sc[:,0,2],'k')

plt.plot(sc.t,a2[:,0,0],'r--',linewidth=2)
plt.plot(sc.t,a2[:,1,1],'g--',linewidth=2)
plt.plot(sc.t,a2[:,2,2],'b--',linewidth=2)
#plt.plot(sc.t,a2[:,0,2],'k--',linewidth=2)

plt.plot(disc.t,disc.a2[:,0,0],'r:')
plt.plot(disc.t,disc.a2[:,1,1],'g:')
plt.plot(disc.t,disc.a2[:,2,2],'b:')
#plt.plot(disc.t,disc.a2[:,0,2],'k:')

print(disc.a2[1,...].diagonal().sum())
print(disc.a2[0,...].diagonal().sum())
