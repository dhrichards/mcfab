#%%
import numpy as np
import shtns
from scipy.special import sph_harm


class BuildHarmonics:
    def __init__(self,x,w,L=6,mmax=6):
        ''' Build spherical harmonic representation from 
        discrete samples on sphere
        $f^l_m = \frac{1}{N}\sum_i^N w_i \bar{Y}^l_m(\theta_i,\phi_i)$
        
        Input:
        x: 3xnpoints array of xyz coordinates
        w: npoints array of mass values
        L: maximum spherical harmonic degree
        mmmax: maximum spherical harmonic order'''

        self.x = x
        self.w = w
        self.L = L
        self.mmax = mmax
        self.sh = shtns.sht(L,self.mmax)
        self.f = self.sh.spec_array()

        self.theta = np.arccos(x[2,:])
        self.phi = np.arctan2(x[1,:],x[0,:])


        for l in range(L+1):
            for m in range(0,min(l,mmax)+1):
                self.f[self.sh.idx(l,m)] = self.sum_conj(l,m)

        self.f /= self.f[0]



    def sum_conj(self,l,m):
        harm_conj = (-1)**m * sph_harm(-m,l,self.phi,self.theta)

        return np.mean(self.w * harm_conj)

    def __call__(self):
        return self.f
    

    def plot(self,hemisphere=False):
        nlats, nlons = self.sh.set_grid(100,100,\
                shtns.SHT_PHI_CONTIGUOUS,1.e-10)
    
        theta_vals = np.arccos(self.sh.cos_theta)
        phi_vals = (2.0*np.pi/nlons)*np.arange(nlons)

        ra,dec = decra_from_polar(phi_vals,theta_vals)


        X, Y = np.meshgrid(ra, dec)

        fgrid = self.sh.synth(self.f)


        if hemisphere:
            

            fgrid = fgrid[0:nlats//2,:]
            X = X[0:nlats//2,:]
            Y = Y[0:nlats//2,:]

        return X,Y,fgrid
    



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






