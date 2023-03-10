#%%
import jax.numpy as np
import matplotlib.pyplot as plt
from meso_fab_mc import meso_mc_jax as mc_jax
from meso_fab_mc import meso_mc as mc_np

# np.random.seed(0)
# gradu = np.random.rand(3,3)
# gradu[2,2] = -gradu[0,0] - gradu[1,1]


gradu = np.array([[ 0.5,  0. ,  0. ],\
                  [ 0. ,  0.5,  0. ],\
                    [ 0. ,  0. , -1. ]])
# gradu[0,2] = 1
#gradu[0,1] = 2

# gradu = np.array([[ 2.62396833e-04,  4.28081245e-04,  0.00000000e+00],
#        [ 1.57583637e-04, -1.80731124e-04,  0.00000000e+00],
#        [ 0.00000000e+00,  0.00000000e+00, -8.16657096e-05]])



dt = 0.1
tmax = 10
T = -5
x = np.array([1,0.1,0])
#x=None
## Discrete

disc_jax = mc_jax.solver(5000)
%timeit disc_jax.solve_constant(gradu,dt,tmax,x)

disc_np = mc_np.solver(5000)
%timeit disc_np.solve_constant(gradu,dt,tmax,x)

plt.plot(disc_jax.t,disc_jax.a2[:,0,0],'r:',label='JAX')
plt.plot(disc_jax.t,disc_jax.a2[:,1,1],'g:')
plt.plot(disc_jax.t,disc_jax.a2[:,2,2],'b:')
#plt.plot(disc.t,disc.a2[:,0,2],'k:')

plt.plot(disc_np.t,disc_np.a2[:,0,0],'r',label='Numpy')
plt.plot(disc_np.t,disc_np.a2[:,1,1],'g')
plt.plot(disc_np.t,disc_np.a2[:,2,2],'b')


plt.ylim(-0.1,1)
plt.legend()

