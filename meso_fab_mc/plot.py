import numpy as np
import matplotlib.pyplot as plt
import meso_fab_mc.meso_mc as mc
import scipy.optimize 
from scipy.special import logsumexp
import cartopy



class SphericalKDE(object):
    """ Spherical kernel density estimator
    Parameters
    ----------
    phi_samples, theta_samples : array_like
        spherical-polar samples to construct the kde
    weights : array_like
        Sample weighting
        default [1] * len(phi_samples))
    bandwidth : float
        bandwidth of the KDE. Increasing bandwidth increases smoothness
    density : int
        number of grid points in theta and phi to draw contours.
    Attributes
    ----------
    phi, theta : np.array
        spherical polar samples
    weights : np.array
        Sample weighting (normalised to sum to 1).
    bandwidth : float
        Bandwidth of the kde. defaults to rule-of-thumb estimator
        https://en.wikipedia.org/wiki/Kernel_density_estimation
        Set to None to use this value
    density : int
        number of grid points in theta and phi to draw contours.
    palefactor : float
        getdist-style colouration factor of sigma-contours.
    """
    def __init__(self, x,
                 weights=None, bandwidth=None, density=100):
        
        # Transform to spherical coords
        self.theta = np.arccos(x[2,:])
        self.phi = np.arctan2(x[1,:],x[0,:])


        if weights is None:
            weights = np.ones_like(self.phi)
        self.weights = np.array(weights) / sum(weights)
        self.bandwidth = bandwidth
        self.density = density
        self.palefactor = 0.6

        if len(self.phi) != len(self.theta):
            raise ValueError("phi_samples must be the same"
                             "shape as theta_samples ({}!={})".format(
                                 len(self.phi), len(self.theta)))
        if len(self.phi) != len(self.weights):
            raise ValueError("phi_samples must be the same"
                             "shape as weights ({}!={})".format(
                                 len(self.phi), len(self.weights)))

        sigmahat = VonMises_std(self.phi, self.theta)
        self.suggested_bandwidth = 0.35*1.06*sigmahat*len(weights)**-0.2

    def __call__(self, theta,phi):
        """ Log-probability density estimate
        Parameters
        ----------
        phi, theta : float or array_like
            Spherical polar coordinate
        Returns
        -------
        float or array_like
            log-probability area density

        """
        # Transform to spherical coords
        # theta = np.arccos(x[2,:])
        # phi = np.arctan2(x[1,:],x[0,:])
        return logsumexp(VMF(phi, theta, self.phi, self.theta, self.bandwidth),
                         axis=-1, b=self.weights)




    def plot_hemisphere(self, ax, **kwargs):
        """ Plot the KDE on an axis.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
        Keywords
        --------
        Any other keywords are passed to `matplotlib.axes.Axes.contourf`
        """
        
        # Compute the kernel density estimate on an equiangular grid
        ra = np.linspace(-180, 180, self.density)
        dec = np.linspace(0, 89, self.density)
        X, Y = np.meshgrid(ra, dec)
        phi, theta = polar_from_decra(X, Y)

        P = np.exp(self(phi, theta))

        ax.gridlines(alpha=0.2)
        # Plot the countours on a suitable equiangular projection
        ax.pcolormesh(X, Y, P,transform=cartopy.crs.PlateCarree(), *kwargs)



    def plot(self, ax, **kwargs):
        """ Plot the KDE on an axis.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
        Keywords
        --------
        Any other keywords are passed to `matplotlib.axes.Axes.contourf`
        """
        
        # Compute the kernel density estimate on an equiangular grid
        ra = np.linspace(-180, 180, self.density)
        dec = np.linspace(-89, 89, self.density)
        X, Y = np.meshgrid(ra, dec)
        phi, theta = polar_from_decra(X, Y)

        P = np.exp(self(phi, theta))

        ax.gridlines(alpha=0.2)
        # Plot the countours on a suitable equiangular projection
        ax.pcolormesh(X, Y, P,transform=cartopy.crs.PlateCarree(), *kwargs)

    def plot_samples(self, ax, nsamples=None, **kwargs):
        """ Plot equally weighted samples on an axis.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            matplotlib axis to plot on. This must be constructed with a
            `cartopy.crs.projection`:
            >>> import cartopy
            >>> import matplotlib.pyplot as plt
            >>> fig = plt.subplots()
            >>> ax = fig.add_subplot(111, projection=cartopy.crs.Mollweide())
        nsamples : int
            Approximate number of samples to plot. Can only thin down to
            this number, not bulk up
        Keywords
        --------
        Any other keywords are passed to `matplotlib.axes.Axes.plot`
        """
        ra, dec = self._samples(nsamples)
        ax.plot(ra, dec, 'k.', transform=cartopy.crs.PlateCarree(), *kwargs)

    @property
    def bandwidth(self):
        if self._bandwidth is None:
            return self.suggested_bandwidth
        else:
            return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    def _samples(self, nsamples=None):
        weights = self.weights / self.weights.max()
        if nsamples is not None:
            weights /= weights.sum()
            weights *= nsamples
        i_ = weights > np.random.rand(len(weights))
        phi = self.phi[i_]
        theta = self.theta[i_]
        ra, dec = decra_from_polar(phi, theta)
        return ra, dec
    
I = np.eye(3)

class ODFfromTensors:
    def __init__(self, a2, a4,density=100):
        self.a2 = a2
        self.a4 = a4

        self.b2 = a2 - I/3
        self.density  = density

        self.b4 = a4 - (np.einsum('ij,kl', I, a2) + np.einsum('ik,jl', I, a2) \
                        + np.einsum('il,jk', I, a2) + np.einsum('jk,il', I, a2) \
                        + np.einsum('jl,ik', I, a2) + np.einsum('kl,ij', I, a2))/7 \
                        + (np.einsum('ij,kl',I,I) + np.einsum('ik,jl',I,I) \
                        + np.einsum('il,jk',I,I))/35



    def __call__(self, phi,theta):
        """ Evaluate the ODF at a given direction """

        # phi, theta are nx x ny arrays
        # Transfrom to cartesian coordinates
        n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

        fij = np.einsum('i...,j...->ij...', n, n)
        fij = fij - np.expand_dims(I/3, axis=(2,3))

        fijkl = np.einsum('i...,j...,k...,l...->ijkl...', n, n, n, n)
        fijkl = fijkl -  (np.einsum('ij,kl...->ijkl...', I, fij) + np.einsum('ik,jl...->ijkl...', I, fij) \
                        + np.einsum('il,jk...->ijkl...', I, fij) + np.einsum('jk,il...->ijkl...', I, fij) \
                        + np.einsum('jl,ik...->ijkl...', I, fij) + np.einsum('kl,ij...->ijkl...', I, fij))/7 \
                        - np.expand_dims((np.einsum('ij,kl->ijkl',I,I) + np.einsum('ik,jl->ijkl',I,I) \
                        + np.einsum('il,jk->ijkl',I,I))/35, axis=(4,5))
        

        odf = 1/(4*np.pi) + (15/(8*np.pi))*np.einsum('ij,ij...->...', self.b2, fij) \
        + (315/(32*np.pi))*np.einsum('ijkl,ijkl...->...', self.b4, fijkl)

        return odf   

    def plot(self,ax, **kwargs):
        ra = np.linspace(-180, 180, self.density)
        dec = np.linspace(-89, 89, self.density)
        X, Y = np.meshgrid(ra, dec)
        phi, theta = polar_from_decra(X, Y)

        P = self(phi, theta)
        ax.gridlines(alpha=0.2)
        # Plot the countours on a suitable equiangular projection
        ax.pcolormesh(X, Y, P,transform=cartopy.crs.PlateCarree(), *kwargs)






def VMF(phi, theta, phi0, theta0, sigma0):
    """ Von-Mises Fisher distribution function.
    Parameters
    ----------
    phi, theta : float or array_like
        Spherical-polar coordinates to evaluate function at.
    phi0, theta0 : float or array-like
        Spherical-polar coordinates of the center of the distribution.
    sigma0 : float
        Width of the distribution.
    Returns
    -------
    float or array_like
        log-probability of the vonmises fisher distribution.
    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
    """
    x = cartesian_from_polar(phi, theta)
    x0 = cartesian_from_polar(phi0, theta0)
    norm = -np.log(4*np.pi*sigma0**2) - logsinh(1./sigma0**2)
    return norm + np.tensordot(x, x0, axes=[[0], [0]])/sigma0**2


def logsinh(x):
    """ log(sinh(x)) """
    return x + np.log(1-np.exp(-2*x)) - np.log(2)






def VonMises_std(phi, theta):
    """ Von-Mises sample standard deviation.
    Parameters
    ----------
    phi, theta : array-like
        Spherical-polar coordinate samples to compute mean from.
    Returns
    -------
        solution for
        ..math:: 1/tanh(x) - 1/x = R,
        where
        ..math:: R = || \sum_i^N x_i || / N
    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
        but re-parameterised for sigma rather than kappa.
    """
    x = cartesian_from_polar(phi, theta)
    S = np.sum(x, axis=-1)
    R = S.dot(S)**0.5/x.shape[-1]

    def f(s):
        return 1/np.tanh(s)-1./s-R

    kappa = scipy.optimize.brentq(f, 1e-8, 1e8)
    sigma = kappa**-0.5
    return sigma



def polar_from_decra(ra, dec):
    """ Convert from spherical polar coordinates to ra and dec.
    Parameters
    ----------
    ra, dec : float or np.array
        Right ascension and declination in degrees.
    Returns
    -------
    phi, theta : float or np.array
        Spherical polar coordinates in radians
    """
    phi = np.mod(ra/180.*np.pi, 2*np.pi)
    theta = np.pi/2-dec/180.*np.pi
    return phi, theta



def polar_from_cartesian(x):
    """ Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    x : array_like
        cartesian coordinates
    Returns
    -------
    phi, theta : float or np.array
        azimuthal and polar angle in radians.
    """
    x = np.array(x)
    r = (x*x).sum(axis=0)**0.5
    x, y, z = x
    theta = np.arccos(z / r)
    phi = np.mod(np.arctan2(y, x), np.pi*2)
    return phi, theta



def cartesian_from_polar(phi, theta):
    """ Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or np.array
        azimuthal and polar angle in radians.
    Returns
    -------
    nhat : np.array
        unit vector(s) in direction (phi, theta).
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

