#%%
import numpy as np
import matplotlib.pyplot as plt
import meso_mc as mc
from tqdm import tqdm
import exactsympy as es


#Load ex.a2 from a file
data = np.load('exa2.npz')
exact_a2 = data['a2']
dt = data['dt']
tmax = data['tmax']
gradu = data['gradu']


# np.random.seed(12312)
# gradu = np.random.rand(3,3)
# gradu[2,2] = -gradu[0,0] - gradu[1,1]
# gradu = 0.5*(gradu + gradu.T) 
# # Make symmetric so no vorticity so we can compare with exact solution


# dt = 0.3
# tmax = 0.9
x = [0.0 , 0.00, 1.0]

## Discrete

disc = mc.solver(50000,1e-2,inital_condition='isotropic')
disc.solve_constant(gradu,dt,tmax,x)


## Exact


# ex = es.gbmexact()
# ex.solve_constant(gradu,dt,tmax,x[2])
# exact_a2 = ex.a2


plt.plot(disc.t,disc.a2[:,0,0],'r:')
plt.plot(disc.t,disc.a2[:,1,1],'g:')
plt.plot(disc.t,disc.a2[:,2,2],'b:')
#plt.plot(disc.t,disc.a2[:,0,2],'k:')

plt.plot(disc.t,exact_a2[:,0,0],'r')
plt.plot(disc.t,exact_a2[:,1,1],'g')
plt.plot(disc.t,exact_a2[:,2,2],'b')
#plt.plot(ex.t,ex.a2[:,0,2],'k-.')

#Save ex.a2 to a file along with the parameters used to generate it
np.savez('exa2.npz',a2=exact_a2,dt=dt,tmax=tmax,gradu=gradu)




