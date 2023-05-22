import jax.numpy as jnp
import jax



def EnhancementsfromGammaBeta(gamma,beta):
    Eca = 1/beta
    Ecc = 3/(4*gamma-1)
    return Ecc,Eca

def GammaBetafromEnhancements(Ecc,Eca):
    gamma = (Ecc+3)/(4*Ecc)
    beta = 1/Eca
    return gamma,beta



def integrate(q,m):
    # Integrates over first axis
    mq = jnp.einsum('p,p...->p...',m,q)
    return jnp.mean(mq,axis=0)





def akcalc(c,m,k=2):
    ''' Calculate kth order orientation tensor
    c: orientation vectors (n,3)
    m: mass (n)'''

    # Repeat c and m with negatives
    c = jnp.concatenate((c,-c),axis=0)
    m = jnp.concatenate((m,m),axis=0)

    # Build k rank outer product
    ck = jnp.ones_like(m)
    for i in range(k):
        ck = jnp.einsum('p...,pj->p...j',ck,c)

    # Calculate tensor
    A = integrate(ck,m)
    return A




def random(npoints):
    # Create points on the unit sphere using fibonacci spiral

    # key = jax.random.PRNGKey(0)
    # n = jax.random.normal(key,shape=(npoints,3))
    # n /= jnp.linalg.norm(n,axis=1, keepdims=True)
     # Golden angle in radians
    golden_angle = jnp.pi * (3 - jnp.sqrt(5))

    # Calculate the step size in the z-axis
    z_step = 2 / npoints

    # Indices array
    indices = jnp.arange(npoints)

    # Calculate z, r, and angle values for all points
    z = -1 + indices * z_step
    r = jnp.sqrt(1 - z ** 2)
    angle = indices * golden_angle

    # Calculate x, y, and z coordinates
    x = r * jnp.cos(angle)
    y = r * jnp.sin(angle)

    # Stack x, y, and z coordinates into a single array
    points = jnp.column_stack((x, y, z))


    # Shuffle points
    key = jax.random.PRNGKey(0)
    points = jax.random.permutation(key,points)

    return points


def single_max(npoints):
    n = jnp.zeros((npoints,3))
    n[:,2] = 1
    return n


def a2calc(n,m):
    # # Calculate 2nd order orientation tensor
    # mninj = jnp.einsum('n,ni,nj->nij',m,n,n)
    # A = jnp.mean(mninj,axis=0)

    # # Symmetrise: this is like assuming we have another set of 
    # # particles with opposite orientation
    # A = 0.5*(A + A.T)
    # Now aliases to akcalc
    A = akcalc(n,m,k=2)
    return A

def a4calc(n,m):
    # # Calculate 4th order orientation tensor
    # mnnnnn = jnp.einsum('p,pi,pj,pk,pl->pijkl',m,n,n,n,n) 
    # A = jnp.mean(mnnnnn,axis=0)

    # # Symmetrise: this is like assuming we have another set of 
    # # particles with opposite orientation
    # A4symm = (1/24)*(jnp.einsum('ijkl->ijkl',A)\
    #                 + jnp.einsum('ijkl->jikl',A)\
    #                 + jnp.einsum('ijkl->ijlk',A)\
    #                 + jnp.einsum('ijkl->jilk',A)\
    #                 + jnp.einsum('ijkl->klij',A)\
    #                 + jnp.einsum('ijkl->lkij',A)\
    #                 + jnp.einsum('ijkl->klji',A)\
    #                 + jnp.einsum('ijkl->lkji',A)\
    #                 + jnp.einsum('ijkl->ikjl',A)\
    #                 + jnp.einsum('ijkl->kijl',A)\
    #                 + jnp.einsum('ijkl->iklj',A)\
    #                 + jnp.einsum('ijkl->kilj',A)\
    #                 + jnp.einsum('ijkl->jlik',A)\
    #                 + jnp.einsum('ijkl->ljik',A)\
    #                 + jnp.einsum('ijkl->jlki',A)\
    #                 + jnp.einsum('ijkl->ljki',A)\
    #                 + jnp.einsum('ijkl->iljk',A)\
    #                 + jnp.einsum('ijkl->lijk',A)\
    #                 + jnp.einsum('ijkl->ilkj',A)\
    #                 + jnp.einsum('ijkl->likj',A)\
    #                 + jnp.einsum('ijkl->jkil',A)\
    #                 + jnp.einsum('ijkl->kjil',A)\
    #                 + jnp.einsum('ijkl->jkli',A)\
    #                 + jnp.einsum('ijkl->kjli',A))
    # Now aliases to akcalc
    A4symm = akcalc(n,m,k=4)

    return A4symm



def StrengthR(n,m):
    # Defined from Castelnau et al. 1996
    sum = m*jnp.sum(n,axis=0)
    N = n.shape[0]

    R = (2 * jnp.linalg.norm(sum,axis=1) -N)/N

    return R

