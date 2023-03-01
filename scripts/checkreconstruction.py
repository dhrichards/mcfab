#%%
# Script to compare shtns and original implementation of speccaf
import numpy as np
import matplotlib.pyplot as plt
from meso_fab_mc import meso_mc as mc
from meso_fab_mc import buildharmonics
from meso_fab_mc import reconstruction
from tqdm import tqdm
import exactsympy as es


gradu = np.zeros((3,3))
gradu[0,0] = -0.5
gradu[1,1] = -0.5
gradu[2,2] = +1

dt = 0.1
tmax = 5
x = [1.0 , 0.1, 0.0]

## Discrete

disc = mc.solver(5000,1e-2)
disc.solve_constant(gradu,dt,tmax,x)

recon = reconstruction.Reconstruct(disc.n,disc.m)

bh = buildharmonics.BuildHarmonics(disc.n,disc.m)
import shtns
from scipy.special import sph_harm
L=6
x = disc.n
w = disc.m
mmax = min(L,6)
sh = shtns.sht(L,mmax)
f = sh.spec_array()

theta = np.arccos(x[2,:])
phi = np.arctan2(x[1,:],x[0,:])

def sum_conj(l,m):
    harm_conj = (-1)**m * sph_harm(-m,l,phi,theta)

    return np.mean(w * harm_conj)

for l in range(L+1):
    for m in range(0,min(l,mmax)+1):
        f[sh.idx(l,m)] = sum_conj(l,m)

f /= f[0]

import cartopy
fig,ax = plt.subplots(1,2,figsize=(10,5))#,subplot_kw=dict(projection=cartopy.crs.AzimuthalEquidistant(central_latitude=0,central_longitude=0)))

X,Y,F = recon.plot(True)
ax[0].pcolormesh(X,Y,F)

X,Y,F = bh.plot(True)
ax[1].pcolormesh(X,Y,F)
# %%
