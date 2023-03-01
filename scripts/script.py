#%%
import shtns
import numpy as np
from matplotlib import pyplot as plt
from speccaf.speccaf_shtns import solver as solver_shtns
import speccaf.spherical as sp
import speccaf.solver as solver

gradu = np.zeros((3,3))
# gradu[0,0] = 0.5
# gradu[1,1] = 0.5
# gradu[2,2] = -1
gradu[0,1] = 2

gradu = gradu*1
T=-25
x = [6.9136665 , 0.78361228, 4.40951074]
sc = solver_shtns(lmax=20,mmax=6)

f_new = sc.solve_constant(gradu,T,dt=0.005,tmax=1.0,x=x)

a2_new = sc.a2(f_new)
#eigvals_new = np.linalg.eigvals(a2_new.real)

# sh = sp.spherical(6)
# f0 = sh.fabricfromdiagonala2(1/3)
# nt = sc.nsteps
# f=np.zeros((sc.nsteps,f0.size),dtype='complex128')
# f[0,:]=f0
# # Calculate fabric tensor
# a2=np.zeros((nt,3,3))
# a4=np.zeros((nt,3,3,3,3))
# a2[0,...] = sh.a2(f[0,:])
# a4[0,...] = sh.a4(f[0,:])
# rk = solver.rk3iterate(T, gradu, sh,x=x)
# strain = np.linspace(0,sc.tmax,nt)
# for i in range(nt-1):


#     #Update fabric with dt T[i] gradu[i]
    
#     f[i+1,:] = rk.iterate(f[i,:], sc.dt)

#     # Update orientation tensors
#     a2[i+1,...] = sh.a2(f[i+1,:])
#     a4[i+1,...] = sh.a4(f[i+1,:])



    
# eigvals = np.linalg.eigvals(a2)

plt.figure()
plt.plot(sc.t,a2_new[:,0,0].real,'r',label='new')
plt.plot(sc.t,a2_new[:,1,1].real,'b',label='new')
plt.plot(sc.t,a2_new[:,2,2].real,'g',label='new')
plt.plot(sc.t,a2_new[:,0,1].real,'k',label='new')
plt.plot(sc.t,a2_new[:,0,2].real,'y',label='new')

# plt.plot(strain,a2[:,0,0].real,'r--',label='old')
# plt.plot(strain,a2[:,1,1].real,'b--',label='old')
# plt.plot(strain,a2[:,2,2].real,'g--',label='old')
# plt.plot(strain,a2[:,0,1].real,'k--',label='old')
# plt.plot(strain,a2[:,0,2].real,'y--',label='old')
# plt.legend()

#%%

p = speccaf_shtns.plotting(sc.sh)
xx,yy,fgrid = p.polefigure(f_new[-1,:])

fig,ax = plt.subplots()

p.plot_polefigure(ax,f_new[-1,:])

#3d plot of fabric
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p.plot_spherical(ax,f_new[-1,:])
