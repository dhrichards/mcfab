import jax.numpy as jnp
import numpy as np
import jax
from .measures import *
from .grainflowlaws import GrainRheo
from .golf import GolfStress
from .orthotropic import Orthotropic


def v_star(n,W,D,S,x,inveta):

    iotaD = x[0]
    iotaS = x[1]

    
    Dn = jnp.einsum('ij,pj->pi',D,n)
    Wn = jnp.einsum('ij,pj->pi',W,n)
    Sn = jnp.einsum('ij,pj->pi',S,n)


    Dnnn = jnp.einsum('kl,pl,pk,pi->pi',D,n,n,n)
    Snnn = jnp.einsum('kl,pl,pk,pi->pi',S,n,n,n)

    Dterm = jnp.einsum('pi,p->pi',Dn - Dnnn,inveta)
    Sterm = jnp.einsum('pi,p->pi',Sn - Snnn,inveta)



    v = Wn -iotaD*Dterm - iotaS*Sterm

   

    return v
    


def Deformability(n,D):


    D2 = jnp.einsum('ij,ji',D,D)

    Dn = jnp.einsum('ij,pj->pi',D,n)
    Dnn = jnp.einsum('ij,pj,pi->p',D,n,n)

    Def = (jnp.einsum('pi,pi->p',Dn,Dn) - Dnn**2)/D2

    return Def

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


def ImprovedEuler(n,m,W,D,S,inveta,x,dt,key):
    """Variation of the improved Euler method for SDE
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)"""
    


    lamb = x[2]
    beta = x[3]

   

    k1 = v_star(n,W,D,S,x[:2],inveta)*dt 
    k2 = v_star(n+k1,W,D,S,x[:2],inveta)*dt 
    
    m1 = m*(1+beta*Deformability(n,D)*dt)
    m2 = m*(1+beta*Deformability(n+k1,D)*dt)
    

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

def effectiveSR(D,S):
    D2 = jnp.einsum('ij,ji',D,D)
    return jnp.sqrt(0.5*D2)

def normC(D,S):
    C2 = jnp.einsum('ij,ji',D+S,D+S)
    return jnp.sqrt(0.5*C2)

def iterate(fabric,p):
    n,m,a2,a4 = fabric
    

    gradu,x,dt,key,flowlaw,sr_type = p
    

    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)

    # Calculate S dependent on flow law
    Ecc = x[4]
    Eca = x[5]
    power = x[6]
    alpha_rheo = x[7]

    funcs = [GrainRheo, GolfStress, Orthotropic]
    S,inveta = jax.lax.switch(flowlaw,funcs,D,Ecc,Eca,n,m,power,alpha_rheo)

    # Choose whether we multiply particles by effective SR or normC (as Elmer does)
    SR = jax.lax.cond(sr_type,effectiveSR,normC,D,S)

    # Multiply lambda and beta by SR
    x = x.at[2:4].set(x[2:4]*SR)

    n,m = ImprovedEuler(n,m,W,D,S,inveta,x[:4],dt,key)
    n,m = add_delete(n,m) 

    a2 = a2calc(n,m)
    a4 = a4calc(n,m)
    
    return (n,m,a2,a4),fabric




def solve(npoints,gradu,dt,x,flowlaw='R1',sr_type='SR'):
    """Solve the SDE using the lax scan function
    npoints is the number of particles
    gradu is the velocity gradient (nsteps,3,3))
    dt is the time step (nsteps,)
    x is the non-dimensional parameter vector (nsteps,6)
    x[0] = iotaD
    x[1] = iotaS
    x[2] = lambda
    x[3] = beta
    x[4] = Ecc
    x[5] = Eca
    x[6] = power
    x[7] = alpha_rheo
    
    flowlaws implemented: grain,ortho,golf
    sr_types implemented: SR, normC"""
    x = jnp.array(x,float)



    m = jnp.ones(npoints)
    n = random(npoints)

    a2 = a2calc(n,m)
    a4 = a4calc(n,m)
    fabric_0 = (n,m,a2,a4)

    if flowlaw == 'grain':
        flowlaws = jnp.zeros_like(dt,int)
    elif flowlaw == 'golf':
        flowlaws = jnp.ones_like(dt,int)
    elif flowlaw == 'ortho':
        flowlaws = jnp.ones_like(dt,int)*2


    if sr_type == 'SR':
        sr_types = jnp.ones_like(dt)
    else:
        sr_types = jnp.zeros_like(dt)


    
    nsteps = gradu.shape[0]

    # Split random key into nsteps keys
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key,nsteps)

    final, fabric = jax.lax.scan(iterate,fabric_0,(gradu,x,dt,keys,flowlaws,sr_types))
    return fabric








