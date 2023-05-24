#%%
import numpy as np
from scipy.special import sph_harm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

class BuildHarmonics:
    def __init__(self,x,w,L=6,mmax=6):
        ''' Build spherical harmonic representation from 
        discrete samples on sphere
        $f^l_m = \frac{1}{N}\sum_i^N w_i \bar{Y}^l_m(\theta_i,\phi_i)$
        
        Input:
        x: nx3 points array of xyz coordinates
        w: npoints array of mass values
        L: maximum spherical harmonic degree
        mmmax: maximum spherical harmonic order'''

        self.x = x
        self.w = w
        self.L = L
        self.mmax = mmax
        self.f = self.spec_array()

        self.theta = np.arccos(x[:,2])
        self.phi = np.arctan2(x[:,1],x[:,0])


        for l in range(L+1):
            for m in range(0,min(l,mmax)+1,1):
                self.f[self.idx(l,m)] = self.sum_conj(l,m)

        self.f /= self.f[0]

    def idx(self,l,m):
        # spherical harmonic index without shtns
        # so we can use it for indexing
        return l**2 + l + m
    
    def spec_array(self):
        # create spectral array without shtns
        return np.zeros((self.L+1)**2,dtype=np.complex128)
    
    def synth(self,f,theta,phi):
        # synthesize spherical harmonics from spectral array
        fgrid = np.zeros_like(theta,dtype=np.complex128)
        for l in range(self.L+1):
            for m in range(0,min(l,self.mmax)+1,1):
                fgrid += f[self.idx(l,m)] * sph_harm(m,l,phi,theta)
        
        return fgrid.real


    def sum_conj(self,l,m):
        harm_conj = (-1)**m * sph_harm(-m,l,self.phi,self.theta)

        return np.mean(self.w * harm_conj)

    def __call__(self):
        return self.f

    

    def grid(self,hemisphere=False):

        nlat = 100
        nlon = 100
        theta_vals = np.linspace(0,np.pi,nlat)
        phi_vals = np.linspace(0,2*np.pi,nlon)

        #make phi contiguous so it includes 2pi
        phi_vals = np.append(phi_vals,2*np.pi)
        phi,theta = np.meshgrid(phi_vals,theta_vals)

    


        ra,dec = decra_from_polar(phi_vals,theta_vals)


        X, Y = np.meshgrid(ra, dec)
        

        fgrid = self.synth(self.f,theta,phi)
                           

        if hemisphere:
            

            fgrid = fgrid[0:nlat//2,:]
            X = X[0:nlat//2,:]
            Y = Y[0:nlat//2,:]

        return X,Y,fgrid
    
    def init_fig(self,ncol=1,nrow=1,figsize=(4,3),hemisphere=True):
        if hemisphere:
            fig,axs = plt.subplots(nrow,ncol,figsize=figsize, \
                subplot_kw={'projection':ccrs.AzimuthalEquidistant(90,90)})
            
        return fig,axs
    
    def plot(self,fig,ax,hemisphere=False,colorbar=True,vmax=None,**kwargs):
        
        
        X,Y,F = self.grid(hemisphere=hemisphere)
        pcol = ax.pcolormesh(X,Y,F,transform=ccrs.PlateCarree(),vmin=0,vmax=vmax)
        pcol.set_edgecolor('face')
        ax.set_aspect('equal')
        ax.axis('off')
        kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), \
                            'xlocs':np.arange(-360,+360,45),\
                                'linewidth':0.5, 'color':'black', 'alpha':0.25, \
                                    'linestyle':'-'}
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(),**kwargs_gridlines)#,xlocs=[s_dir,s_dir+90,s_dir+180,s_dir+270])


        if hemisphere:
            gl.ylim = (0,90)

        geo = ccrs.RotatedPole()

        # colorbar for this axes -show max and min
        if colorbar:
            cb = fig.colorbar(pcol,ax=ax,orientation='horizontal',**kwargs)
            cb.set_label('ODF')
            vm = np.max(pcol.get_clim()[1])
            cb.set_ticks([0,vm/2,vm])
            # set sig fig in colorbar
            from matplotlib.ticker import FormatStrFormatter
            cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        return pcol
    
    def J(self):
        J=0
        Sff = 0
        for l in range(0,self.L+1,2):
            Sff = 0*Sff
            for m in range(0,l+1,1):
                Sff=Sff+np.abs(self.f[self.idx(l,abs(m))])**2
            J=J+Sff
        return J
    
    def M(self):

        return M
    

    def y0(self):
        X,Y,F = self.grid()
        # return at y =0
        ind = np.argmin(np.abs(Y[:,0]))
        return F[ind,:]


    

# def odf_from_tensors(a2,a4):
#     sh = shtns.sht(4,4)
#     nlats, nlons = sh.set_grid(100,100,\
#                 shtns.SHT_PHI_CONTIGUOUS,1.e-10)
    
#     theta_vals = np.arccos(sh.cos_theta)
#     phi_vals = (2.0*np.pi/nlons)*np.arange(nlons)

#     ra,dec = decra_from_polar(phi_vals,theta_vals)
#     X, Y = np.meshgrid(ra, dec)
#     Ph, Th = np.meshgrid(phi_vals, theta_vals)


#     b2,b4 = deviatoric_tensors(a2,a4)
#     fij,fijkl = basisfunctions(Ph,Th)

#     odf = 1/(4*np.pi) + (15/(8*np.pi))*np.einsum('ij,ij...->...',b2,fij) \
#         + (315/(32*np.pi))*np.einsum('ijkl,ijkl...->...',b4,fijkl)
    
#     return X,Y,odf
    


    

# def deviatoric_tensors(a2,a4):

#     I = np.eye(3)

#     b2 = a2 - I/3

#     b4 = a4 - (1/7)*(np.einsum('ij,kl->ijkl',I,a2) +
#                     np.einsum('ik,jl->ijkl',I,a2) +
#                     np.einsum('il,jk->ijkl',I,a2) +
#                     np.einsum('jk,il->ijkl',I,a2) +
#                     np.einsum('jl,ik->ijkl',I,a2) +
#                     np.einsum('kl,ij->ijkl',I,a2)) \
#             + (1/35)*(np.einsum('ij,kl->ijkl',I,I) + np.einsum('ik,jl->ijkl',I,I) + np.einsum('il,jk->ijkl',I,I))
    
#     return b2,b4
    

# def basisfunctions(ph,th):
#     x = np.sin(th)*np.cos(ph)
#     y = np.sin(th)*np.sin(ph)
#     z = np.cos(th)
#     n = np.array([x,y,z])

#     I = np.eye(3)

#     #fij = ninj
#     fij = np.einsum('i,...,j,...->ij,...',n,n) - np.eye(3)[...,None,None]/3


#     fijkl = np.einsum('i,...,j,...,k,...,l,...->ijkl,...',n,n,n,n) \
#         - (1/7)*(np.einsum('ij,k...,l...->ijkl...',I,n,n) + 
#                 np.einsum('ik,j...,l...->ijkl...',I,n,n) +
#                 np.einsum('il,j...,k...->ijkl...',I,n,n) +
#                 np.einsum('jk,i...,l...->ijkl...',I,n,n) +
#                 np.einsum('jl,i...,k...->ijkl...',I,n,n) +
#                 np.einsum('kl,i...,j...->ijkl...',I,n,n))\
#         + (1/35)*(np.einsum('ij,kl->ijkl',I,I) + np.einsum('ik,jl->ijkl',I,I) + np.einsum('il,jk->ijkl',I,I))
    

#     return fij,fijkl



def decra_from_polar(phi, theta):
    """ Convert from ra and dec to spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians
    Returns
    -------
    ra, dec : float or numpy.array
        Right ascension and declination in degrees.
    """
    ra = phi * (phi < np.pi) + (phi-2*np.pi)*(phi > np.pi)
    dec = np.pi/2-theta
    return ra/np.pi*180, dec/np.pi*180






