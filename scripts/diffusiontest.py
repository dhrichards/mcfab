#%%
import numpy as np
import matplotlib.pyplot as plt
import meso_fab_mc as mc
from tqdm import tqdm


np.random.seed(0)
gradu = np.random.rand(3,3)
gradu[2,2] = -gradu[0,0] - gradu[1,1]

# gradu = np.zeros((3,3))
# gradu[0,0] = 0.5
# gradu[1,1] = 0.5
# gradu[2,2] = -1
# gradu[0,2] = 1
# gradu[0,1] = 1

dt = 0.01
tmax = 20
x = [0.0 , 0.01, 0.0]

## Discrete

disc = mc.solver(10000,1e-2,inital_condition='single_max')
disc.solve_constant(gradu,dt,tmax,x,integrator='ForwardEuler')


# Plotting

plt.plot(disc.t,disc.a2[:,0,0],'r:')
plt.plot(disc.t,disc.a2[:,2,2],'b:')



#Plot exact solution from a2 evolution equation
#dA/dt = lambda*(delta_ij - 3A_ij)

A00 = (1-np.exp(-3*x[1]*disc.t))/3
A22 = (1+2*np.exp(-3*x[1]*disc.t))/3

plt.plot(disc.t,A00,'r')
plt.plot(disc.t,A22,'b')