#%%
import numpy as np
import matplotlib.pyplot as plt
import meso_fab_mc.meso_mc as mc
import meso_fab_mc.plot as plot
import cartopy.crs as ccrs
import spherical_kde as skde
from scipy.stats import gaussian_kde

gradu = np.zeros((3,3))
gradu[0,0] = -0.5
gradu[1,1] = -0.5
gradu[2,2] = +1


dt = 1
tmax = 200
x = [1.0 , 0.1, 1.0]

## Discrete

disc = mc.solver(10000)
disc.solve_constant(gradu,dt,tmax,x)


theta,phi = plot.polar_from_cartesian(disc.n)

kde = plot.SphericalKDE(theta,phi,weights=disc.m,bandwidth=0.35)


fig,ax = plt.subplots(subplot_kw=dict(projection=ccrs.Robinson()))

kde.plot(ax)


fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))

ax.scatter(disc.n[0,:],disc.n[1,:],disc.n[2,:],s=disc.m)


print(disc.a2[-1,...])

plt.figure()
plt.hist(np.log10(disc.m),bins=50)

mass_kde = gaussian_kde(np.log10(disc.m))

# %%
