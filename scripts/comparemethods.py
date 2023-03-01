#%%
# Script to compare shtns and original implementation of speccaf
import numpy as np
import matplotlib.pyplot as plt
from meso_fab_mc import meso_mc as mc
from meso_fab_mc import BuildHarmonics



# np.random.seed(0)
# gradu = np.random.rand(3,3)
# gradu[2,2] = -gradu[0,0] - gradu[1,1]

gradu = np.zeros((3,3))
gradu[0,0] = 0.5
gradu[1,1] = 0.5
gradu[2,2] = -1
# gradu[0,2] = 1
#gradu[0,1] = 1

# gradu = 1e4*np.array([[ 2.62396833e-04,  4.28081245e-04,  0.00000000e+00],
#        [ 1.57583637e-04, -1.80731124e-04,  0.00000000e+00],
#        [ 0.00000000e+00,  0.00000000e+00, -8.16657096e-05]])




T = -10

dt = 0.5
tmax = 10
x = [1.0 , 0.0, 0.0]
npoints =5000


## Discrete

st = mc.solver(npoints)
st.solve_constant(gradu,dt,tmax,x,method='Static')

tay = mc.solver(npoints)
tay.solve_constant(gradu,dt,tmax,x,method='Taylor')

c = mc.solver(npoints)
c.solve_constant(gradu,dt/50,tmax,x,method='C',alpha=0.04)


plt.plot(st.t,st.a2[:,0,0],'r:')
plt.plot(st.t,st.a2[:,1,1],'g:')
plt.plot(st.t,st.a2[:,2,2],'b:')
plt.plot(st.t,st.a2[:,0,2],'k:')

plt.plot(tay.t,tay.a2[:,0,0],'r')
plt.plot(tay.t,tay.a2[:,1,1],'g')
plt.plot(tay.t,tay.a2[:,2,2],'b')
plt.plot(tay.t,tay.a2[:,0,2],'k')

plt.plot(c.t,c.a2[:,0,0],'r--')
plt.plot(c.t,c.a2[:,1,1],'g--')
plt.plot(c.t,c.a2[:,2,2],'b--')
plt.plot(c.t,c.a2[:,0,2],'k--')



import cartopy
tay.odf = BuildHarmonics(tay.n,tay.m)
st.odf = BuildHarmonics(st.n,st.m)
c.odf = BuildHarmonics(c.n,c.m)

def plot(ax,odf,hemisphere=True):
    X,Y,F = odf.plot(hemisphere=hemisphere)
    pcol = ax.pcolormesh(X,Y,F,transform=cartopy.crs.PlateCarree())
    pcol.set_edgecolor('face')
    ax.set_aspect('equal')
    ax.axis('off')
    gl = ax.gridlines(alpha=0.3)
    if hemisphere:
        gl.ylim = (0,90)

    return pcol

fig,ax = plt.subplots(1,3,figsize=(10,5),\
            subplot_kw={'projection':cartopy.crs.Orthographic(0,45)})
plot(ax=ax[0],odf=tay.odf,hemisphere=False)
plot(ax=ax[1],odf=st.odf,hemisphere=False)
plot(ax=ax[2],odf=c.odf,hemisphere=False)


