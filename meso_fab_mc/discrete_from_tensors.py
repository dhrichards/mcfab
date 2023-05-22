import jax.numpy as jnp
from .static_mc import a2calc,a4calc,random
import jax.scipy.optimize as jopt

def build_discrete(a2,a4,npoints=10000):

    # Get initial guess
    n = random(npoints)
    theta = jnp.arccos(n[:,2])
    phi = jnp.arctan2(n[:,1],n[:,0])
    thph = jnp.concatenate((theta,phi))

    # Minimize
    res = jopt.minimize(error_fun,thph,args=(a2,a4),method='BFGS',options={'maxiter':5000})

    # Get final result
    theta = res.x[:npoints]
    phi = res.x[npoints:]
    n = jnp.stack((jnp.sin(theta)*jnp.cos(phi),jnp.sin(theta)*jnp.sin(phi),jnp.cos(theta)),axis=1)
    m = jnp.ones(npoints)

    return n,m




def error_fun(thph,a2,a4):
    # thph = (theta,phi)
    # a2 = (3,3)
    # a4 = (3,3,3,3)
    npoints = thph.size//2
    theta = thph[:npoints]
    phi = thph[npoints:]
    n = jnp.stack((jnp.sin(theta)*jnp.cos(phi),jnp.sin(theta)*jnp.sin(phi),jnp.cos(theta)),axis=1)
    m = jnp.ones(npoints)

    a2_d = a2calc(n,m)
    a4_d = a4calc(n,m)

    return residual(a2_d,a4_d,a2,a4)


def residual(a2_d,a4_d,a2,a4):
    #From GOLF paper
    Del2 = jnp.abs(a2_d - a2)
    Del4 = jnp.abs(a4_d - a4)

    return jnp.einsum('ij,ji',Del2,Del2) + jnp.einsum('ijkl,lkji',Del4,Del4)