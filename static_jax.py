
import jax.numpy as jnp

def xi_calc(gamma,beta,eta):
        xi0 = beta/(2*eta)
        xi1 = (gamma + 1)/(4*gamma -1) -1/beta
        xi2 = 1/beta -1
        xi3 = -(2/3)*((gamma+2)/(4*gamma-1)-1)

        return jnp.array([xi0,xi1,xi2,xi3])


def fluidity(A2,A4,gamma,beta,eta):
    # Fluidity Tensor 
    I = jnp.eye(3)
    xi = xi_calc(gamma,beta,eta)

    F = xi[0]*(jnp.einsum('ik,jl->ijkl',I,I) \
        + 2*xi[1]*A4 \
        + xi[2]*(jnp.einsum('ik,lj->ijkl',I,A2) + jnp.einsum('ik,jl->ijkl',A2,I)) \
        + xi[3]*jnp.einsum('kl,ij->ijkl',A2,I))
    return F
    

def fluidity_star(n,gamma,beta,eta):
    # Fluidity* tensor

    I = jnp.eye(3)
    xi = xi_calc(gamma,beta,eta)
    # does this need masses?
    F = xi[0]*(jnp.einsum('ik,jl->ijkl',I,I) \
        + 2*xi[1]*jnp.einsum('pi,pj,pk,pl->pijkl'n,n,n,n) \
        + xi[2]*(jnp.einsum('ik,pl,pj->pijkl',I,n,n) \
                 + jnp.einsum('pi,pk,jl->pijkl',n,n,I)) \
        + xi[3]*jnp.einsum('pk,pl,ij->pijkl',n,n,I))
    return F

    

def Scalc(D,A2,A4,gamma,beta,eta):
    # Stress tensor
     
    F = fluidity(A2,A4,gamma,beta,eta)
    S = jnp.linalg.tensorsolve(F,D)

    return S
    

def Ccalc(D,A2,A4,gamma=1,beta=0.04,eta=1,alpha=0.04):
    # C based on Gillet-Chaulet 2006
    S = Scalc(D,A2,A4,gamma,beta,eta)

    C = (1-alpha)*D + alpha*S/(2*eta)
    return C[jnp.newaxis,:,:]

def Dstarcalc(n,D,A2,A4,gamma=1,beta=0.04,eta=1):
     
    S = Scalc(D,A2,A4,gamma,beta,eta)
    Fstar = fluidity_star(n,gamma,beta,eta)
    Dstar = jnp.einsum('pijkl,kl->pij',Fstar,S)

    return Dstar


