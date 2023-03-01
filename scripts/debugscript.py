#%%
import numpy as np
import shtns

lmax = 20
mmax = 4
        
sh = shtns.sht(lmax,mmax)

nlats, nlons = sh.set_grid()
theta_vals = np.arccos(sh.cos_theta)
phi_vals = (2.0*np.pi/nlons)*np.arange(nlons)
phi, theta = np.meshgrid(phi_vals, theta_vals)

x = np.sin(theta)*np.cos(phi)
y = np.sin(theta)*np.sin(phi)
z = np.cos(theta)

n = np.array([x,y,z])

f0 = sh.spec_array()
f0[0] = 1.0




def v_star_cal(gradu):
    gradu = gradu
    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)
    D2 = np.einsum('ij,ji',D,D)
    effectiveSR = np.sqrt(0.5*D2)

    Dn = np.einsum('ij,jpq->ipq',D,n)
    Wn = np.einsum('ij,jpq->ipq',W,n)
    Dnn = np.einsum('ij,jpq,ipq->pq',D,n,n)

    v = Wn -iota*(Dn - Dnn*n)
    Dstar = 5*(np.einsum('ipq,ipq->pq',Dn,Dn) - Dnn**2)/D2
    

    

def vec_cart2sph(v):
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
    # return in harmonic from f*(D^* - <D^*>)
    f_spat = sh.synth(f)
    
    fDstar = sh.analys(f_spat*Dstar) # take advantage of spherical harmonic integration
    GBM_spat = f_spat*(Dstar - fDstar[0].real)
    return sh.analys(GBM_spat) 


def RHS(f):
    #Right hand side of the ODE
    f_spat = sh.synth(f)
    fvth, fvph = vec_cart2sph(f_spat*v)
    divfv = div(fvth, fvph)
    return -divfv + effectiveSR*lamb*lap(f) +effectiveSR*beta*GBM(f)

def RK4(f,dt):
    #4th order Runge-Kutta
    k1 = RHS(f)
    k2 = RHS(f + 0.5*dt*k1)
    k3 = RHS(f + 0.5*dt*k2)
    k4 = RHS(f + dt*k3)
    return f + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def params(T):
    #Params from Richards et al. 2021, normalised to effective strain rate
    iota = 0.0259733*T + 1.95268104
    lamb = 0.00251776*T + 0.41244777
    beta = 0.35182521*T + 12.17066493


def solve_constant(gradu,T,dt=0.01,tmax=1.0,x=None):
    dt = dt
    tmax = tmax
    t  = np.arange(0,tmax,dt)
    nsteps = len(t)

    if x is not None:
        iota = x[0]
        lamb = x[1]
        beta = x[2]
    else:
        params(T)
    

    f = np.zeros((nsteps,sh.nlm),dtype=np.complex128)
    f[0,...] = f0

    
    
    v_star_cal(gradu)
    for i in range(nsteps-1):
        f[i+1,...] = RK4(f[i,...],dt)

    return f

def iterate(f,T,gradu,dt,x=None):


    if x is not None:
        iota = x[0]
        lamb = x[1]
        beta = x[2]
    else:
        params(T)
    
    v_star_cal(gradu)
    f = RK4(f,dt)
    return f


def a2(f):
    if f.ndim==1:
        f = f.expand_dims(0)
    a=np.zeros((f.shape[0],3,3),dtype=complex) 

    a[:,0,0]=((0.33333333333333333333)+(0)*1j)*f[:,sh.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[:,sh.idx(2,0)]+ 2*((0.18257418583505537115)+(0)*1j)*f[:,sh.idx(2,2)]
    a[:,0,1]=2*((0)+(0.18257418583505537115)*1j)*f[:,sh.idx(2,2)]
    a[:,1,0]=a[:,0,1]
    a[:,0,2]=2*((-0.18257418583505537115)+(0)*1j)*f[:,sh.idx(2,1)]
    a[:,2,0]=a[:,0,2]
    a[:,1,1]=((0.33333333333333333333)+(0)*1j)*f[:,sh.idx(0,0)]+ ((-0.14907119849998597976)+(0)*1j)*f[:,sh.idx(2,0)]+ 2*((-0.18257418583505537115)+(0)*1j)*f[:,sh.idx(2,2)]
    a[:,1,2]=2*((0)+(-0.18257418583505537115)*1j)*f[:,sh.idx(2,1)]
    a[:,2,1]=a[:,1,2]
    a[:,2,2]=((0.33333333333333333333)+(0)*1j)*f[:,sh.idx(0,0)]+ ((0.29814239699997195952)+(0)*1j)*f[:,sh.idx(2,0)]
    
    a.squeeze()
    return a

x = [6.9136665 , 0.78361228, 4.40951074]
iota = x[0]
lamb = x[1]
beta = x[2]


gradu = np.array([[0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]],)
    


f = solve_constant(gradu,0.0,dt=0.01,tmax=2,x=x)

a = a2(f)

from matplotlib import pyplot as plt

plt.plot(a[:,0,0].real)
plt.plot(a[:,1,1].real)
plt.plot(a[:,2,2].real)