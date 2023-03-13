#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from meso_fab_mc import meso_mc_jax as mc_jax
from meso_fab_mc import meso_mc as mc_np

# jnp.random.seed(0)
# gradu = jnp.random.rand(3,3)
# gradu[2,2] = -gradu[0,0] - gradu[1,1]


gradu = np.array([[ 0.5,  0. ,  1. ],\
                  [ 0.6 ,  0.5,  0. ],\
                    [ 1. ,  0. , -1. ]])
# gradu[0,2] = 1
#gradu[0,1] = 2

# gradu = jnp.array([[ 2.62396833e-04,  4.28081245e-04,  0.00000000e+00],
#        [ 1.57583637e-04, -1.80731124e-04,  0.00000000e+00],
#        [ 0.00000000e+00,  0.00000000e+00, -8.16657096e-05]])



dt = 0.1
tmax = 5
T = -5
x = jnp.array([1,0.1,2])
#x=None
## Discrete
t,nsteps = mc_jax.time(dt,tmax)
gradu_tile,dt_tile,x_tile = mc_jax.tile_arrays(gradu,dt,tmax,x)
#gradu_tile = gradu_tile.at[:,2,2].set(jnp.linspace(0,0,nsteps))

n,m = mc_jax.solve(5000,gradu_tile,dt_tile,x_tile)

a2 = mc_jax.a2calc(n,m)


disc_np = mc_np.solver(5000)
disc_np.solve_constant(gradu,dt,tmax,x)

disc_np2 = mc_np.solver(5000)


a2_np = np.zeros((nsteps,3,3))
a4_np = np.zeros((nsteps,3,3,3,3))

n = np.zeros((nsteps,5000,3))
m = np.zeros((nsteps,5000))

a2_np[0,...] = mc_np.a2calc(disc_np2.n)
a4_np[0,...] = mc_np.a4calc(disc_np2.n)

n[0,...] = disc_np.n
m[0,...] = disc_np.m

for i in range(nsteps-1):

    disc_np2.iterate(gradu_tile[i,...],dt,x)

    n[i+1,...] = disc_np2.n
    m[i+1,...] = disc_np2.m 
    a2_np[i+1,...] = mc_np.a2calc(disc_np2.n,disc_np2.m)
    a4_np[i+1,...] = mc_np.a4calc(disc_np2.n,disc_np2.m)



plt.plot(t,a2[:,0,0],'r:',label='JAX')
plt.plot(t,a2[:,1,1],'g:')
plt.plot(t,a2[:,2,2],'b:')
#plt.plot(disc.t,disc.a2[:,0,2],'k:')

plt.plot(t,a2_np[:,0,0],'r',label='Numpy')
plt.plot(t,a2_np[:,1,1],'g')
plt.plot(t,a2_np[:,2,2],'b')

plt.plot(t,disc_np.a2[:,0,0],'r--',label='Numpy Constant')
plt.plot(t,disc_np.a2[:,1,1],'g--')
plt.plot(t,disc_np.a2[:,2,2],'b--')

plt.ylim(-0.1,1)
plt.legend()

