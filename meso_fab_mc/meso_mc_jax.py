import jax.numpy as jnp
import numpy as np
import jax
from .static_jax import * 



def gradu_measures(gradu):
    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)
    D2 = jnp.einsum('ij,ji',D,D)
    effectiveSR = jnp.sqrt(0.5*D2)

    return D,W,D2,effectiveSR


def v_star(n,Dstar,W,iota):

    
    Dn = jnp.einsum('pij,pj->pi',Dstar,n)
    Wn = jnp.einsum('ij,pj->pi',W,n)

    Dnnn = jnp.einsum('pkl,pl,pk,pi->pi',Dstar,n,n,n)
    

    v = Wn -iota*(Dn - Dnnn)

    return v
    
    
def V_star(n,Dstar,W,iota):


    Dnn = jnp.einsum('pij,pj,pi->p',Dstar,n,n)
    

    V = W - iota*\
        (Dstar - jnp.einsum('p,ij->pij',Dnn,jnp.eye(3)))
        
    
    return V




def GBM(n,m,D):


    D2 = jnp.einsum('pij,pji->p',D,D)

    Dn = jnp.einsum('pij,pj->pi',D,n)
    Dnn = jnp.einsum('pij,pj,pi->p',D,n,n)

    Def = 5*(jnp.einsum('pi,pi->p',Dn,Dn) - Dnn**2)/D2

    # a2 = a2calc(n,m)
    # a4 = a4calc(n,m)

    return Def #- 5*(jnp.einsum('ij,ik,kj',D,D,a2) -\
                #        jnp.einsum('ij,kl,ijkl',D,D,a4))/D2


def weiner(n,lamb,dt,key):
    """
    Add Brownian motion to a set of unit vectors on the surface of a sphere
    by adding normally distributed noise to the x, y, and z coordinates,
    and then re-normalizing the vectors.
    
    vectors: numpy array, initial unit vectors (shape: (n, 3))
    dt: float, time step
    sigma: float, standard deviation of the Gaussian distribution
    """
    sigma = jnp.sqrt(2*lamb*dt)

    noise = jax.random.normal(key, shape=(n.shape))*sigma

    n2 = noise + n
    norms = jnp.linalg.norm(n2, axis=1, keepdims=True)
    n2  = n2/norms
    return n2 - n


def add_delete(n,m):
    # n shape npoints,3
    # m shape npoints
    npoints = n.shape[0]
    
    #Sort by m
    sort = jnp.argsort(m)
    n = n[sort]
    m = m[sort]

    #Split into low and high mass groups
    n_low = n[:npoints//2]
    n_high = n[npoints//2:]

    m_low = m[:npoints//2]
    m_high = m[npoints//2:]

    # Flip high mass group to go from largest to smallest
    n_high = n_high[::-1]
    m_high = m_high[::-1]

    # Create soft mask for particles to delete
    tol = 0.01
    delete_mask = jnp.where(m_low<tol,1,0)
    #delete mask is 1 for particles to delete, 0 for particles to keep

    #Update positions
    n_low = ((1-delete_mask)*n_low.T + delete_mask*n_high.T).T

    #Update masses
    m_high_new = m_high - delete_mask*m_high/2
    m_low_new = (1-delete_mask)*m_low + delete_mask*m_high/2 


    #Combine low and high mass groups
    n = jnp.concatenate((n_low,n_high))
    m = jnp.concatenate((m_low_new,m_high_new))

    # Normalize masses
    m /= jnp.mean(m)

    return n,m



def ImprovedEuler(n,m,W,Dstar,x,dt,key):
    """Variation of the improved Euler method for SDE
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)"""
    

    iota = x[0]
    lamb = x[1]
    beta = x[2]


    k1 = v_star(n,Dstar,W,iota)*dt 
    k2 = v_star(n+k1,Dstar,W,iota)*dt 
    
    m1 = m*(1+beta*GBM(n,m,Dstar)*dt)
    m2 = m*(1+beta*GBM(n+k1,m,Dstar)*dt)
    

    n = n + 0.5*(k1+k2) + weiner(n,lamb,dt,key)
    m = 0.5*(m1+m2)

    n = n/jnp.linalg.norm(n,axis=1,keepdims=True)
    m /= jnp.mean(m)

    return n,m


def time(dt,tmax):
    t  = jnp.arange(0,tmax,dt)
    nsteps = len(t)
    return t,nsteps

def tile_arrays(gradu,dt,tmax,x):

    t,nsteps = time(dt,tmax)
    # Tile arrays to be nsteps long
    gradu_tile = jnp.tile(gradu,(nsteps,1,1))
    dt_tile = jnp.tile(dt,(nsteps,))
    x_tile = jnp.tile(x,(nsteps,1))

    return gradu_tile,dt_tile,x_tile

def effectiveSR(D,Dstar):
    D2 = jnp.einsum('ij,ji',D,D)
    return jnp.sqrt(0.5*D2)

def normC(D,C):
    C = C[0,...]
    return jnp.sqrt(jnp.einsum('ij,ji->',C,C))

def iterate(fabric,p):
    n,m,a2,a4 = fabric

    gradu,x,dt,key,mode,sr_type = p
    

    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)

    # Choose whether we use C or full orientation dependent Dstar
    Dstar = jax.lax.cond(mode,Ccalc,Dstarcalc,n,D,a2,a4,x[3],x[4])

    # Choose whether we multiply particles by effective SR or normC (as Elmer does)
    SR = jax.lax.cond(sr_type,effectiveSR,normC,D,Dstar)

    # Multiply lambda and beta by SR
    x = x.at[1:3].set(x[1:3]*SR)

    n,m = ImprovedEuler(n,m,W,Dstar,x[:3],dt,key)
    n,m = add_delete(n,m)

    a2 = a2calc(n,m)
    a4 = a4calc(n,m)
    
    return (n,m,a2,a4),fabric




def solve(npoints,gradu,dt,x,model='C',sr_type='SR'):
    """Solve the SDE using the lax scan function
    npoints is the number of particles
    gradu is the velocity gradient (nsteps,3,3))
    dt is the time step (nsteps,)
    x is the non-dimensional parameter vector (nsteps,3)"""

    m = jnp.ones(npoints)
    n = random(npoints)

    a2 = a2calc(n,m)
    a4 = a4calc(n,m)
    fabric_0 = (n,m,a2,a4)

    if model == 'C':
        modes = np.ones_like(dt)
    else:
        modes = np.zeros_like(dt)

    if sr_type == 'SR':
        sr_types = np.ones_like(dt)
    else:
        sr_types = np.zeros_like(dt)


    
    nsteps = gradu.shape[0]

    # Split random key into nsteps keys
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key,nsteps)

    final, fabric = jax.lax.scan(iterate,fabric_0,(gradu,x,dt,keys,modes,sr_types))
    return fabric



def Richards2021(T):
    T[T<-30] = -30
    iota = 0.0259733*T + 1.95268104
    lambtilde = (0.00251776*T + 0.41244777)
    betatilde = (0.35182521*T + 12.17066493)
    alpha = jnp.zeros_like(T)
    ks = jnp.ones_like(T)

    x = jnp.array([iota,lambtilde,betatilde,alpha,ks])
    
    return x.T

def Elmer(T):
    iota = jnp.ones_like(T)
    lambtilde = 2e-3 * jnp.exp(jnp.log(10)*T/10)
    betatilde = jnp.zeros_like(T)
    alpha = 0.06*jnp.ones_like(T)
    ks = 10*jnp.ones_like(T)

    x = jnp.array([iota,lambtilde,betatilde,alpha,ks])
    
    return x.T




def random(npoints):
    # Create n random points on the unit sphere

    key = jax.random.PRNGKey(0)
    n = jax.random.normal(key,shape=(npoints,3))
    n /= jnp.linalg.norm(n,axis=1, keepdims=True)
    return n

def isotropic(npoints):
    # Create 100 sets of random points on spheres
    # and calculate the variance of the diagonal of the 2nd order tensor
    # away from 1/3, choose best one
    key = jax.random.PRNGKey(0)
    n = jax.random.normal(key,shape=(npoints,3,100))
    norm = jnp.linalg.norm(n,axis=1, keepdims=True)
    n /= norm

    a2 = a2calc(n)
    std_dev = jnp.zeros(100)

    a2diag = jnp.diag(a2)

    std_dev = jnp.std(a2diag)

    best = jnp.argmin(std_dev)

    return n[:,:,best]

def single_max(npoints):
    n = jnp.zeros((npoints,3))
    n[:,2] = 1
    return n


def a2calc(n,m=1):
    # Calculate 2nd order orientation tensor
    mninj = jnp.einsum('...n,...ni,...nj->...ijn',m,n,n)
    A = jnp.mean(mninj,axis=-1)
    return A

def a4calc(n,m=1):
    # Calculate 4th order orientation tensor
    mnnnnn = jnp.einsum('...p,...pi,...pj,...pk,...pl->...ijklp',m,n,n,n,n)
    return jnp.mean(mnnnnn,axis=-1)


def StrengthR(n,m):
    # Defined from Castelnau et al. 1996
    sum = m*jnp.sum(n,axis=0)
    N = n.shape[0]

    R = (2 * jnp.linalg.norm(sum,axis=1) -N)/N

    return R






