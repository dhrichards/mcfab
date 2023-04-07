import jax.numpy as jnp
import golf_jax

etaI = golf_jax.load_viscosity_data()

def xi_calc(gamma,beta,eta):
        # xi0 = beta/2
        # xi1 = 2*((gamma + 2)/(4*gamma -1) -1/beta)
        # xi2 = 1/beta -1
        # xi3 = -(1/3)*(xi1 + 2*xi2)

        xi0 = 1.0/2
        xi1 = 2*(beta*(gamma + 2)/(4*gamma -1) -1)
        xi2 = 1 - beta
        xi3 = -(1/3)*(xi1 + 2*xi2)

        return jnp.array([xi0,xi1,xi2,xi3])


def fluidity(A2,A4,gamma,beta,eta):
    # Fluidity Tensor 
    I = jnp.eye(3)
    xi = xi_calc(gamma,beta,eta)

    F = xi[0]*(beta*jnp.einsum('ik,jl->ijkl',I,I) \
        + xi[1]*A4 \
        + xi[2]*(jnp.einsum('ik,lj->ijkl',I,A2) + jnp.einsum('ik,jl->ijkl',A2,I)) \
        + xi[3]*jnp.einsum('kl,ij->ijkl',A2,I))
    return F
    

def fluidity_star(n,gamma,beta,eta):
    # Fluidity* tensor

    I = jnp.eye(3)
    xi = xi_calc(gamma,beta,eta)
    # does this need masses?
    F = xi[0]*(beta*jnp.einsum('ik,jl->ijkl',I,I) \
        + xi[1]*jnp.einsum('pi,pj,pk,pl->pijkl',n,n,n,n) \
        + xi[2]*(jnp.einsum('ik,pl,pj->pijkl',I,n,n) \
                 + jnp.einsum('pi,pk,jl->pijkl',n,n,I)) \
        + xi[3]*jnp.einsum('pk,pl,ij->pijkl',n,n,I))
    return F

    

def Scalc(D,A2,A4,gamma,beta,eta):
    # Stress tensor
     
    F = fluidity(A2,A4,gamma,beta,eta)
    S = jnp.linalg.tensorsolve(F,D)

    return S

def CcalcGOLF(n,D,A2,A4,alpha,ks=1,gamma=1,beta=0.04,eta=1):
    # C based on Gillet-Chaulet 2006

    S = golf_jax.GolfStress(A2,D,etaI)

    C = (1-alpha)*D + ks*alpha*S/(2*eta)
    return jnp.tile(C,(n.shape[0],1,1))
    

def Ccalc(n,D,A2,A4,alpha,ks=1,gamma=1,beta=0.04,eta=1):
    # C based on Gillet-Chaulet 2006
    S = Scalc(D,A2,A4,gamma,beta,eta)

    C = (1-alpha)*D + ks*alpha*S/(2*eta)
    return jnp.tile(C,(n.shape[0],1,1))

def Dstarcalc(n,D,A2,A4,alpha,ks=1,gamma=1,beta=0.04,eta=1):
     
    S = Scalc(D,A2,A4,gamma,beta,eta)
    Fstar = fluidity_star(n,gamma,beta,eta)
    Dstar = jnp.einsum('pijkl,kl->pij',Fstar,S)

    C = (1-alpha)*DTaylor(n,D,A2,A4) + ks*alpha*Dstar

    return C

def DTaylor(n,D,A2,A4):
     return jnp.tile(D,(n.shape[0],1,1))


