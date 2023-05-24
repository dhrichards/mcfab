import jax.numpy as jnp
from .measures import *
from .grainflowlaws import Fluidity



def Orthotropic(D,Ecc,Eca,c,mf,power):

    alpha = 0.0125 # hard-coded value from Rathmann et al. 2022

    a2 = akcalc(c,mf,2)
    a4 = akcalc(c,mf,4)


    _,m = jnp.linalg.eigh(a2)

    E = Eij_tranisotropic(a2,a4,Ecc,Eca,alpha)

    S = jnp.zeros((3, 3))
    lamb = jnp.zeros(6)
    I = jnp.zeros(6)
    gamma = 0
    eta = 0

    j = jnp.array([1, 2, 0])
    k = jnp.array([2, 0, 1])

    I = I.at[0].set(jnp.einsum('ij,ji',D,(jnp.outer(m[:,1],m[:,1]) - jnp.outer(m[:,2],m[:,2]))/2))
    I = I.at[1].set(jnp.einsum('ij,ji',D,(jnp.outer(m[:,2],m[:,2]) - jnp.outer(m[:,0],m[:,0]))/2))
    I = I.at[2].set(jnp.einsum('ij,ji',D,(jnp.outer(m[:,0],m[:,0]) - jnp.outer(m[:,1],m[:,1]))/2))
    I = I.at[3].set(jnp.einsum('ij,ji',D,(jnp.outer(m[:,1],m[:,2]) + jnp.outer(m[:,2],m[:,1]))/2))
    I = I.at[4].set(jnp.einsum('ij,ji',D,(jnp.outer(m[:,2],m[:,0]) + jnp.outer(m[:,0],m[:,2]))/2))
    I = I.at[5].set(jnp.einsum('ij,ji',D,(jnp.outer(m[:,0],m[:,1]) + jnp.outer(m[:,1],m[:,0]))/2))

    for i in range(3):
        p = 2/(power+1)
        lamb = lamb.at[i].set((4/3)*(E[j[i],j[i]]**p + E[k[i],k[i]]**p - E[i,i]**p))
        lamb = lamb.at[i+3].set(2*E[j[i],k[i]]**p)
        gamma += 2*(E[j[i],j[i]]*E[k[i],k[i]])**p - E[i,i]**(2*p)



    for i in range(3):
        S += (lamb[i]/gamma)*(I[j[i]]-I[k[i]])*(jnp.eye(3)-3*jnp.outer(m[:,i],m[:,i]))/2\
            +(4/lamb[i+3])*I[i+3]*(jnp.outer(m[:,j[i]],m[:,k[i]])+jnp.outer(m[:,k[i]],m[:,j[i]]))/2
        
        eta += (lamb[i]/gamma)*(I[j[i]]-I[k[i]])**2\
            +(4/lamb[i+3])*I[i+3]**2
    
    eta = eta**((1-power)/(2*power))

    inveta = jnp.ones_like(mf)/eta

    return S,inveta



def Eij_tranisotropic(a2,a4,Ecc,Eca,alpha):
    _,v = jnp.linalg.eigh(a2)
    e1 = v[:,0]
    e2 = v[:,1]
    e3 = v[:,2]


    E1 = Evw_tranisotropic(e1,e1,tau_vv(e1),a2,a4,Ecc,Eca,alpha)
    E2 = Evw_tranisotropic(e2,e2,tau_vv(e2),a2,a4,Ecc,Eca,alpha)
    E3 = Evw_tranisotropic(e3,e3,tau_vv(e3),a2,a4,Ecc,Eca,alpha)

    E4 = Evw_tranisotropic(e2,e3,tau_vw(e2,e3),a2,a4,Ecc,Eca,alpha)
    E5 = Evw_tranisotropic(e1,e3,tau_vw(e1,e3),a2,a4,Ecc,Eca,alpha)
    E6 = Evw_tranisotropic(e1,e2,tau_vw(e1,e2),a2,a4,Ecc,Eca,alpha)

    Eij = jnp.array([[E1,E6,E5],\
                     [E6,E2,E4],\
                     [E5,E4,E3]])

    return Eij



def Evw_tranisotropic(v,w,tau,a2,a4,Ecc,Eca,alpha):

    vw = jnp.einsum('i,j->ij',v,w)

    Evw_sachs = jnp.einsum('ij,ji',sachshomo(tau,a2,a4,Ecc,Eca),vw)/\
                jnp.einsum('ij,ji',sachshomo(tau,*isotropictensors(),Ecc,Eca),vw)
    Evw_taylor = jnp.einsum('ij,ji',taylorhomo(tau,a2,a4,Ecc,Eca),vw)/\
                jnp.einsum('ij,ji',taylorhomo(tau,*isotropictensors(),Ecc,Eca),vw)

    Evw = (1-alpha)*Evw_sachs + alpha*Evw_taylor

    return Evw


def tau_vv(v):
    return jnp.eye(3)/3 - jnp.einsum('i,j->ij',v,v)

def tau_vw(v,w):
    return jnp.einsum('i,j->ij',v,w) + jnp.einsum('i,j->ij',w,v)


def sachshomo(tau,a2,a4,Ecc,Eca):
    F = Fluidity(Ecc,Eca,1.0,a2,a4)
    return jnp.einsum('ijkl,kl->ij',F,tau)

def taylorhomo(tau,a2,a4,Ecc,Eca):
    mu = TaylorViscosity(Ecc,Eca,a2,a4)
    return jnp.linalg.tensorsolve(mu,tau)

def TaylorViscosity(Ecc,Eca,a2,a4):
    return Fluidity(1/Ecc,1/Eca,1.0,a2,a4)

def isotropictensors():
    a2 = jnp.eye(3)/3
    a4 = (1/15)*(jnp.einsum('ij,kl->ijkl',jnp.eye(3),jnp.eye(3)) \
            + jnp.einsum('ik,jl->ijkl',jnp.eye(3),jnp.eye(3)) \
            + jnp.einsum('il,jk->ijkl',jnp.eye(3),jnp.eye(3)))
    return a2,a4