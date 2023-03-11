import jax.numpy as jnp
import numpy as np
import jax
import .static_jax as static 



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


    D2 = jnp.einsum('ij,ji',D,D)

    Dn = jnp.einsum('ij,pj->pi',D,n)
    Dnn = jnp.einsum('ij,pj,pi->p',D,n,n)

    Def = 5*(jnp.einsum('pi,pi->p',Dn,Dn) - Dnn**2)/D2

    a2 = a2calc(n,m)
    a4 = a4calc(n,m)

    return Def - 5*(jnp.einsum('ij,ik,kj',D,D,a2) -\
                        jnp.einsum('ij,kl,ijkl',D,D,a4))/D2


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



def ImprovedEuler(n,m,gradu,x,dt,key):
    """Variation of the improved Euler method for SDE
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)"""
    
    D,W,D2,effectiveSR = gradu_measures(gradu)
    Dstar = D[jnp.newaxis,:,:]

    iota = x[0]
    lamb = x[1]*effectiveSR
    beta = x[2]*effectiveSR


    k1 = v_star(n,Dstar,W,iota)*dt 
    k2 = v_star(n+k1,Dstar,W,iota)*dt 
    
    m1 = m*(1+beta*GBM(n,m,D)*dt)
    m2 = m*(1+beta*GBM(n+k1,m,D)*dt)
    

    n = n + 0.5*(k1+k2) + weiner(n,lamb,dt,key)
    m = 0.5*(m1+m2)

    n = n/jnp.linalg.norm(n,axis=1,keepdims=True)
    m /= jnp.mean(m)

    return n,m





def time(dt,tmax):
    t  = jnp.arange(0,tmax,dt)
    nsteps = len(t)
    return t,nsteps


def solve_constant(npoints,gradu,dt,tmax,x):

    m = jnp.ones(npoints)
    n = random(npoints)
    
    t,nsteps = time(dt,tmax)

    init_val = (n,m,gradu,x,dt)


    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key,nsteps)

    # for i in range(1,nsteps):
            
    #         n,m = ImprovedEuler(n,m,gradu,x,dt)
            
    #         #n,m = add_delete(n,m)

    def iterate(prev_val,key):
        n,m,gradu,x,dt = prev_val

        n,m = ImprovedEuler(n,m,gradu,x,dt,key)
        n,m = add_delete(n,m)
        
        return (n,m,gradu,x,dt),prev_val
    

    final, vals = jax.lax.scan(iterate,init_val,keys)

    n = vals[0]
    m = vals[1]


    return n,m


    # def iterate(i,val):

    #     n = val[0]
    #     m = val[1]
    #     a2 = val[2]
    #     a4 = val[3]
    #     gradu = val[4]
    #     x = val[5]
    #     dt = val[6]

    #     n,m = ImprovedEuler(n,m,gradu,x,dt)
        
    #     #n,m = add_delete(n,m)

    #     a2 = a2.at[i,:,:].set(a2calc(n,m))
    #     a4 = a4.at[i,:,:].set(a4calc(n,m))

    #     return (n,m,a2,a4,gradu,x,dt)


    # val = jax.lax.fori_loop(1,nsteps,iterate,\
    #                   (n,m,a2,a4,gradu,x,dt))
    
    # n = val[0]
    # m = val[1]
    # a2 = val[2]
    # a4 = val[3]



        


def params(T,effectiveSR):


    iota = 0.0259733*T + 1.95268104
    lambtilde = (0.00251776*T + 0.41244777)
    betatilde = (0.35182521*T + 12.17066493)
    
    lamb = lambtilde*effectiveSR
    beta = betatilde*effectiveSR

    return jnp.array([iota,lamb,beta])



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





