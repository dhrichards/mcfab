import jax.numpy as jnp
import jaxopt
import jax
from .measures import *




def GrainRheo(D,Ecc,Eca,c,m,power,alpha):

    S_sachs,inveta_sachs = Sachs(D,Ecc,Eca,c,m,power)
    S_taylor,inveta_taylor = Taylor(D,Ecc,Eca,c,m,power)

    S = (1-alpha)*S_sachs + alpha*S_taylor
    inveta = (1-alpha)*inveta_sachs + alpha*inveta_taylor
    
    return S,inveta

def Sachs(D,Ecc,Eca,c,m,power):
    return jax.lax.cond(power==1, Sachs1, Sachs3, D,Ecc,Eca,c,m)



def Taylor(D,Ecc,Eca,c,m,power):
    return jax.lax.cond(power==1, Taylor1, Taylor3, D,Ecc,Eca,c,m)


def Sachs1(D,Ecc,Eca,c,m):
    '''Rathmann Linear'''
    nc0 = akcalc(c,m,0)
    nc2 = akcalc(c,m,2)
    nc4 = akcalc(c,m,4)


    F = Fluidity(Ecc,Eca,nc0,nc2,nc4)
    S = jnp.linalg.tensorsolve(F,D)

    # Normalise
    hatS = (0.4*Eca + 0.2*Ecc + 0.4)*S

    inveta=jnp.ones_like(m)

    return hatS,inveta

def Sachs3(D,Ecc,Eca,c,m):

    
    def Stox(S):
        return jnp.array([S[0,0],S[1,1],0.5*(S[1,2]+S[2,1])\
                          ,0.5*(S[2,0]+S[0,2]),0.5*(S[1,0]+S[0,1])])

    def xtoS(x):
        S = jnp.array([[x[0],x[4],x[3]],\
                       [x[4],x[1],x[2]],
                       [x[3],x[2],-(x[1]+x[0])]])
        return S
    
    # Initial guess
    S0,_ = Sachs1(D,Ecc,Eca,c,m)
    x0 = Stox(S0)

    def objective(x,D,Ecc,Eca,c,m):
        S = xtoS(x)
        # Correction based on errata for n=3 Ecc -> Ecc^(2/(n+1)) etc.
        Ecc = jnp.sqrt(Ecc)
        Eca = jnp.sqrt(Eca)

        nc0 = int_inveta_ckn3(S,Ecc,Eca,c,m,0)
        nc2 = int_inveta_ckn3(S,Ecc,Eca,c,m,2)
        nc4 = int_inveta_ckn3(S,Ecc,Eca,c,m,4)

        F = Fluidity(Ecc,Eca,nc0,nc2,nc4)

        errorS = D - jnp.einsum('ijkl,kl->ij',F,S)
        return Stox(errorS)

    it = jaxopt.LevenbergMarquardt(objective, tol=1e-5, maxiter=10)
    x = it.run(x0,D=D,Ecc=Ecc,Eca=Eca,c=c,m=m)
    S = xtoS(x.params)

    etam1 = inveta(S,Ecc,Eca,c,m)

    return S,etam1


def Taylor1(D,Ecc,Eca,c,m):
    nc0 = akcalc(c,m,0)
    nc2 = akcalc(c,m,2)
    nc4 = akcalc(c,m,4)

    mu = Fluidity(1/Ecc,1/Eca,nc0,nc2,nc4)
    S = jnp.einsum('ijkl,kl->ij',mu,D)

    inveta=jnp.ones_like(m)

    #todo: normalise S

    return S,inveta

def Taylor3(D,Ecc,Eca,c,m):
    #TODO
    S = jnp.zeros((3,3))
    inveta=jnp.ones_like(m)
    return S,inveta


def Fluidity(Ecc,Eca,nc0,nc2,nc4):

    p1,p2,p3 = flowlawparams(Ecc,Eca)
    I = jnp.eye(3)

    
    F = nc0*jnp.einsum('ik,jl->ijkl',I,I) \
        + p1*jnp.einsum('ij,kl->ijkl',I,nc2) \
        + p2*jnp.einsum('ijkl->ijkl',nc4) \
        + p3*(jnp.einsum('ik,lj->ijkl',I,nc2) + jnp.einsum('ik,jl->ijkl',nc2,I))
    
    return F

def ks(Ecc,Eca):
    k1 = (3*(Ecc-1) - 4*(Eca-1))/2
    k2 = 2*(Eca-1)
    return [k1,k2]



def inveta(s,Ecc,Eca,c,n):
    ''' Calculate $\eta^-1$ for different valus of n'''

    # Errata correction
    Ecc = Ecc**(2/(n+1))
    Eca = Eca**(2/(n+1))

    k = ks(Ecc,Eca)

    inveta = jnp.einsum('ij,ji',s,s) \
        + k[0]*jnp.einsum('ij,ni,nj->n',s,c,c)**2 \
        + k[1]*jnp.einsum('ik,kj,ni,nj->n',s,s,c,c)
    
    return inveta**((n-1)/2)


def int_inveta_ckn3(s,Ecc,Eca,c,m,k):
    #For n=3
    ak = akcalc(c,m,k)
    akp2 = akcalc(c,m,k+2)
    akp4 = akcalc(c,m,k+4)

    k = ks(Ecc,Eca)

    term1 = jnp.einsum('ij,ji',s,s)*ak
    term2 = k[0]*jnp.einsum('ij,ji...kl,lk->...',s,akp4,s)
    term3 = k[1]*jnp.einsum('ik,kj,ji...->...',s,s,akp2)

    return (term1 + term2 + term3)


def flowlawparams(Ecc,Eca):
    p1 = -(Ecc-1)/2
    p2 = (3*(Ecc-1) - 4*(Eca-1))/2
    p3 = Eca -1

    return p1,p2,p3







def Sachs1a2a4(D,Ecc,Eca,a2,a4):

    F = Fluidity(Ecc,Eca,1.0,a2,a4)
    S = jnp.linalg.tensorsolve(F,D)

    # Normalise
    hatS = (0.4*Eca + 0.2*Ecc + 0.4)*S
    return hatS
    
def Sachs3FWD(S,Ecc,Eca,c,m):

    # Correction based on errata for n=3 Ecc -> Ecc^(2/(n+1)) etc.
    Ecc = jnp.sqrt(Ecc)
    Eca = jnp.sqrt(Eca)

    nc0 = int_inveta_ckn3(S,Ecc,Eca,c,m,0)
    nc2 = int_inveta_ckn3(S,Ecc,Eca,c,m,2)
    nc4 = int_inveta_ckn3(S,Ecc,Eca,c,m,4)

    F = Fluidity(Ecc,Eca,nc0,nc2,nc4)

    D = jnp.einsum('ijkl,kl->ij',F,S)

    return D

