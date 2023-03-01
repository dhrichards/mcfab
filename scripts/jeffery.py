#%%
import shtns
import numpy as np
from matplotlib import pyplot as plt

gradu = np.zeros((3,3))
# gradu[0,0] = 0.5
# gradu[1,1] = 0.5
# gradu[2,2] = -1
gradu[0,2] = 1
gradu = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, -1]])

iota=1
lamb = 0.01
beta = 2



D = 0.5*(gradu + gradu.T)
W = 0.5*(gradu - gradu.T)

lmax = 50
mmax=4
sh = shtns.sht(lmax, mmax)
nlats, nlons = sh.set_grid()
theta_vals = np.arccos(sh.cos_theta)
phi_vals = (2.0*np.pi/nlons)*np.arange(nlons)
phi, theta = np.meshgrid(phi_vals, theta_vals)

x = np.sin(theta)*np.cos(phi)
y = np.sin(theta)*np.sin(phi)
z = np.cos(theta)

n = np.array([x,y,z])

Dn = np.einsum('ij,jpq->ipq',D,n)
Wn = np.einsum('ij,jpq->ipq',W,n)

Dnn = np.einsum('ij,jpq,ipq->pq',D,n,n)

v = Wn -iota*(Dn - Dnn*n)

# v = np.einsum('ij,jpq->ipq', W, n) - iota*(np.einsum('ij,jpq->ipq', D, n) - \
#     np.einsum('jk,kpq,jpq,ipq->ipq', D, n, n, n))

Dstar = 5*(np.einsum('ipq,ipq->pq',Dn,Dn) - Dnn**2)/np.einsum('mn,nm',D,D)


def vec_cart2sph(v,theta,phi):
    vtheta = v[0,...]*np.cos(theta)*np.cos(phi) + v[1,...]*np.cos(theta)*np.sin(phi) - v[2,...]*np.sin(theta)
    vphi = -v[0,...]*np.sin(phi) + v[1,...]*np.cos(phi)
    return vtheta, vphi



def div(vtheta,vphi):
    #Divergence of vector field on unit sphere, returned in spherical harmonics
    #See https://en.wikipedia.org/wiki/Vector_spherical_harmonics#Divergence
    S,T = sh.analys(vtheta, vphi)
    return -sh.l*(sh.l+1)*S

def lap(f):
    #Laplacian of spherical harmonic array, returned in spherical harmonics
    #See https://en.wikipedia.org/wiki/Spherical_harmonics#Laplacian
    return -sh.l*(sh.l+1)*f

def GBM(f):
    f_spat = sh.synth(f)
    
    fDstar = sh.analys(f_spat*Dstar) # take advantage of spherical harmonic integration
    GBM_spat = f_spat*(Dstar - fDstar[0].real)
    return sh.analys(GBM_spat) 


f = sh.spec_array()
f[0] = 1.0
f_spat = sh.synth(f)

fvth, fvph = vec_cart2sph(f_spat*v, theta, phi)

divfv = div(fvth, fvph)

def RHS(f):
    #Right hand side of the ODE
    f_spat = sh.synth(f)
    fvth, fvph = vec_cart2sph(f_spat*v, theta, phi)
    divfv = div(fvth, fvph)
    return -divfv + lamb*lap(f) +beta*GBM(f)

def RK4(f, dt):
    #4th order Runge-Kutta
    k1 = RHS(f)
    k2 = RHS(f + 0.5*dt*k1)
    k3 = RHS(f + 0.5*dt*k2)
    k4 = RHS(f + dt*k3)
    return f + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0


def linear(f):
    return lamb*lap(f)

def nonlinear(f):
    f_spat = sh.synth(f)
    fvth, fvph = vec_cart2sph(f_spat*v, theta, phi)
    divfv = div(fvth, fvph)
    return -divfv + beta*GBM(f)

def SemiImplicitEuler(f, dt):
    #Semi-implicit Euler with diffusion term implicit
    RH = f + dt*nonlinear(f)
    return RH/(1 + dt*lamb*sh.l*(sh.l+1))


def a2(f):

    a=np.zeros((3,3),dtype=complex)  
    a[0,0]=((0.33333333333333333333)+(0)*1j)*f[sh.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[sh.idx(2,0)]+ 2*((0.18257418583505537115)+(0)*1j)*f[sh.idx(2,2)]
    a[0,1]=2*((0)+(0.18257418583505537115)*1j)*f[sh.idx(2,2)]
    a[1,0]=a[0,1]
    a[0,2]=2*((-0.18257418583505537115)+(0)*1j)*f[sh.idx(2,1)]
    a[2,0]=a[0,2]
    a[1,1]=((0.33333333333333333333)+(0)*1j)*f[sh.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[sh.idx(2,0)]+ 2*((-0.18257418583505537115)+(0)*1j)*f[sh.idx(2,2)]
    a[1,2]=2*((0)+(-0.18257418583505537115)*1j)*f[sh.idx(2,1)]
    a[2,1]=a[1,2]
    a[2,2]=((0.33333333333333333333)+(0)*1j)*f[sh.idx(0,0)]+ ((0.29814239699997195952)+(0)*1j)*f[sh.idx(2,0)]
    
    #a=a*sqrt(4*pi)
    return a

dt = 0.08
tmax = 10.0
t = np.arange(0.0, tmax, dt)
nt = len(t)
a = np.zeros((nt,3,3),dtype=complex)
for i in range(nt):
    f = SemiImplicitEuler(f, dt)
    a[i] = a2(f)



eigvals = np.linalg.eigvals(a)

plt.figure()
plt.plot(t, eigvals.real)


f_spat = sh.synth(f)
fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot_surface(x,y,z, rstride=1, cstride=1, facecolors=plt.cm.viridis(f_spat), linewidth=0, antialiased=False, shade=False)

test = sh.synth(divfv)
test_ = sh.analys(test)

print(np.abs(test_-divfv).max())

