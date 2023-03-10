import jax.numpy as jnp
import numpy as np
import jax
from .static import Static

class solver:
    def __init__(self,npoints,tol=1e-2):
        self.npoints = npoints
        self.m = jnp.ones(npoints)
        self.tol = tol

        self.key = jax.random.PRNGKey(0)

        self.n = self.random()
        



    def gradu_measures(self,gradu):
        D = 0.5*(gradu + gradu.T)
        W = 0.5*(gradu - gradu.T)
        D2 = jnp.einsum('ij,ji',D,D)
        effectiveSR = jnp.sqrt(0.5*D2)

        return D,W,D2,effectiveSR

        


    def v_star(self,n,Dstar,W,iota):

        
        Dn = jnp.einsum('pij,pj->pi',Dstar,n)
        Wn = jnp.einsum('ij,pj->pi',W,n)

        Dnnn = jnp.einsum('pkl,pl,pk,pi->pi',Dstar,n,n,n)
        

        v = Wn -iota*(Dn - Dnnn)

        return v
    
    
    def V_star(self,n,Dstar,W,iota):


        Dnn = jnp.einsum('pij,pj,pi->p',Dstar,n,n)
        

        V = W - iota*\
            (Dstar - jnp.einsum('p,ij->pij',Dnn,jnp.eye(3)))
            
       
        return V




    def GBM(self,n,m,D):


        D2 = jnp.einsum('ij,ji',D,D)

        Dn = jnp.einsum('ij,pj->pi',D,n)
        Dnn = jnp.einsum('ij,pj,pi->p',D,n,n)

        Def = 5*(jnp.einsum('pi,pi->p',Dn,Dn) - Dnn**2)/D2

        a2 = a2calc(n,m)
        a4 = a4calc(n,m)

        return Def - 5*(jnp.einsum('ij,ik,kj',D,D,a2) -\
                         jnp.einsum('ij,kl,ijkl',D,D,a4))/D2
    

    def weiner(self,n,lamb,dt):
        """
        Add Brownian motion to a set of unit vectors on the surface of a sphere
        by adding normally distributed noise to the x, y, and z coordinates,
        and then re-normalizing the vectors.
        
        vectors: numpy array, initial unit vectors (shape: (n, 3))
        dt: float, time step
        sigma: float, standard deviation of the Gaussian distribution
        """
        sigma = jnp.sqrt(2*lamb*dt)

        key,subkey = jax.random.split(self.key)
        self.key = subkey
        noise = jax.random.normal(self.key, shape=(n.shape))*sigma

        n2 = noise + n
        norms = jnp.linalg.norm(n2, axis=1, keepdims=True)
        n2  = n2/norms
        return n2 - n
    

    def add_delete(self,n,m):
        # Add and delete particles:
        # If particle has a mass less than tol, delete it
        # For each deleted particle, split the largest particle into two



        delete = jnp.where(m<0.01)[0]
        n_to_delete = len(delete)

        # # Delete particles
        # n = jnp.delete(n,delete,axis=0)
        # m = jnp.delete(m,delete)

        # Find largest n particles
        largest = jnp.argsort(m)[::-1][:n_to_delete]

        # Split largest particles into ones marked for deletion
        n = n.at[delete,:].set(n[largest,:])
        
        m = m.at[largest].divide(2)
        m = m.at[delete].set(m[largest])

        ## Normalise masses
        m /= jnp.mean(m)

        return n,m


    def ImprovedEuler(self,n,m,gradu,x,dt):
        """Variation of the improved Euler method for SDE
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)"""
        
        D,W,D2,effectiveSR = self.gradu_measures(gradu)
        Dstar = D[jnp.newaxis,:,:]

        iota = x[0]
        lamb = x[1]*effectiveSR
        beta = x[2]*effectiveSR


        k1 = self.v_star(n,Dstar,W,iota)*dt 
        k2 = self.v_star(n+k1,Dstar,W,iota)*dt 
        
        m1 = m*(1+beta*self.GBM(n,m,D)*dt)
        m2 = m*(1+beta*self.GBM(n+k1,m,D)*dt)
        

        n = n + 0.5*(k1+k2) + self.weiner(n,lamb,dt)
        m = 0.5*(m1+m2)

        n = n/jnp.linalg.norm(n,axis=1,keepdims=True)
        m /= jnp.mean(m)

        return n,m

    






    def solve_constant(self,gradu,dt,tmax,x):
        self.t  = jnp.arange(0,tmax,dt)
        self.nsteps = len(self.t)
        self.dt = dt
       
        self.a2 = jnp.zeros((self.nsteps,3,3))
        self.a4 = jnp.zeros((self.nsteps,3,3,3,3))


        a2 = a2calc(self.n,self.m)

        
        a4 = a4calc(self.n,self.m)

        self.a2 = self.a2.at[0,:,:].set(a2)
        self.a4 = self.a4.at[0,:,:].set(a4)

        for i in range(1,self.nsteps):
                
                self.n,self.m = self.ImprovedEuler(self.n,self.m,gradu,x,dt)
                
                #self.n,self.m = self.add_delete(self.n,self.m)
    
                self.a2 = self.a2.at[i,:,:].set(a2calc(self.n,self.m))
                self.a4 = self.a4.at[i,:,:].set(a4calc(self.n,self.m))

        # def iterate(i,val):

        #     n = val[0]
        #     m = val[1]
        #     a2 = val[2]
        #     a4 = val[3]
        #     gradu = val[4]
        #     x = val[5]
        #     dt = val[6]

        #     n,m = self.ImprovedEuler(n,m,gradu,x,dt)
            
        #     #n,m = self.add_delete(n,m)

        #     a2 = a2.at[i,:,:].set(a2calc(self.n,self.m))
        #     a4 = a4.at[i,:,:].set(a4calc(self.n,self.m))

        #     return (n,m,a2,a4,gradu,x,dt)


        # val = jax.lax.fori_loop(1,self.nsteps,iterate,\
        #                   (self.n,self.m,self.a2,self.a4,gradu,x,dt))
        
        # self.n = val[0]
        # self.m = val[1]
        # self.a2 = val[2]
        # self.a4 = val[3]



        


    def params(self,T):
        self.iota = 0.0259733*T + 1.95268104
        self.lambtilde = (0.00251776*T + 0.41244777)
        self.betatilde = (0.35182521*T + 12.17066493)
        
        self.lamb = self.lambtilde*self.effectiveSR
        self.beta = self.betatilde*self.effectiveSR

    
            


    
  


    def random(self):
        # Create n random points on the unit sphere

        key,subkey = jax.random.split(self.key)
        self.key = subkey
        n = jax.random.normal(subkey,shape=(self.npoints,3))
        n /= jnp.linalg.norm(n,axis=1, keepdims=True)


        return n
    
    def isotropic(self):
        # Create 100 sets of random points on spheres
        # and calculate the variance of the diagonal of the 2nd order tensor
        # away from 1/3, choose best one
        key,subkey = jax.random.split(self.key)
        self.key = subkey
        n = jax.random.normal(subkey,shape=(self.npoints,3,100))
        norm = jnp.linalg.norm(n,axis=1, keepdims=True)
        n /= norm

        a2 = jnp.zeros((100,3,3))
        std_dev = jnp.zeros(100)
        for i in range(100):
            a2[i,...] = a2calc(n[:,:,i])
            std_dev[i] = jnp.std(jnp.diag(a2[i,...]))

        
        best = jnp.argmin(std_dev)

        return n[:,:,best]




def a2calc_vec(n,m=1):
    ns = n.shape[2]
    A = jnp.zeros((ns,3,3))

    ninj = jnp.einsum('ipq,jpq->qijp',n,n)
    for i in range(3):
        for j in range(3):
            A[:,i,j] = jnp.mean(m*ninj[i,j,:])

    return A

def a2calc(n,m=1):
    # Calculate 2nd order orientation tensor
    ninj = jnp.einsum('ni,nj->ijn',n,n)
    A = jnp.mean(m*ninj,axis=2)
    return A

def a4calc(n,m=1):
    # Calculate 4th order orientation tensor
    nnnnn = jnp.einsum('pi,pj,pk,pl->ijklp',n,n,n,n)
    return jnp.mean(m*nnnnn,axis=4)





def single_max(npoints):
    n = jnp.zeros((npoints,3))
    n[:,2] = 1
    return n


