import numpy as np
from .static import Static

class solver:
    def __init__(self,npoints,tol=1e-2,inital_condition='isotropic'):
        self.npoints = npoints
        self.m = np.ones(npoints)
        self.tol = tol

        if inital_condition == 'isotropic':
            self.n = isotropic(npoints)
        elif inital_condition == 'random':
            self.n = random(npoints)
        elif inital_condition == 'single_max':
            self.n = single_max(npoints)





    def gradu_measures(self,gradu):
        self.gradu = gradu
        self.D = 0.5*(gradu + gradu.T)
        self.W = 0.5*(gradu - gradu.T)
        self.D2 = np.einsum('ij,ji',self.D,self.D)
        self.effectiveSR = np.sqrt(0.5*self.D2)



    def v_star(self,n):
        
        Dn = np.einsum('ij,jp->ip',self.D,n)
        Wn = np.einsum('ij,jp->ip',self.W,n)

        Dnn = np.einsum('ij,jp,ip->p',self.D,n,n)
        

        v = Wn -self.iota*(Dn - Dnn*n)

        return v
    
    def v_star_static(self,n,a2,a4):


        Dstar = self.sachs(a2,a4,self.gradu,n)

        Dn = np.einsum('pij,jp->ip',Dstar,n)
        Wn = np.einsum('ij,jp->ip',self.W,n)

        Dnn = np.einsum('pij,jp,ip->p',Dstar,n,n)
        

        v = Wn -self.iota*(Dn - Dnn*n)

        return v








    def GBM(self,n,m):


        Dn = np.einsum('ij,jp->ip',self.D,n)

        Dnn = np.einsum('ij,jp,ip->p',self.D,n,n)
        Def = 5*(np.einsum('ip,ip->p',Dn,Dn) - Dnn**2)/self.D2

        a2 = a2calc(n,m)
        a4 = a4calc(n,m)
        return Def - 5*(np.einsum('ij,ik,kj',self.D,self.D,a2) - np.einsum('ij,kl,ijkl',self.D,self.D,a4))/self.D2


    def ImprovedEuler(self,n,m,dt):
        """Variation of the improved Euler method for SDE
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)"""
        
        k1 = self.v_star(n)*dt 
        k2 = self.v_star(n+k1)*dt 
        
        m1 = m*(1+self.beta*self.GBM(n,m)*dt)
        m2 = m*(1+self.beta*self.GBM(n+k1,m)*dt)

        n = n + 0.5*(k1+k2) + + self.weiner(n,dt,np.sqrt(self.lamb))
        m = 0.5*(m1+m2)

        n = n/np.linalg.norm(n,axis=0)
        m /= np.mean(m)

        return n,m
    

    def StaticImprovedEuler(self,n,m,dt,a2,a4):
        """Variation of the improved Euler method for SDE
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)"""


        
        k1 = self.v_star_static(n,a2,a4)*dt 
        k2 = self.v_star_static(n+k1,a2,a4)*dt 
        
        m1 = m*(1+self.beta*self.GBM(n,m)*dt)
        m2 = m*(1+self.beta*self.GBM(n+k1,m)*dt)

        n = n + 0.5*(k1+k2) + + self.weiner(n,dt,np.sqrt(self.lamb))
        m = 0.5*(m1+m2)

        n = n/np.linalg.norm(n,axis=0)
        m /= np.mean(m)

        return n,m



    def ForwardEuler(self,n,m,dt):
        # Forward Euler method for SDE
        n += self.v_star(n)*dt + self.weiner(n,dt,np.sqrt(self.lamb))
        m = m*(1+self.beta*self.GBM(n,m)*dt)

        n = n/np.linalg.norm(n,axis=0)
        m /= np.mean(m)

        return n,m


    def iterate(self,gradu,dt,x):
        self.gradu_measures(gradu)
        self.iota = x[0]
        self.lamb = x[1]*self.effectiveSR
        self.beta = x[2]*self.effectiveSR

        self.n,self.m = self.ImprovedEuler(self.n,self.m,dt)
        self.add_delete()






    def solve_constant(self,gradu,dt,tmax,x,method='ImprovedEuler'):
        self.t  = np.arange(0,tmax,dt)
        self.nsteps = len(self.t)

        self.gradu_measures(gradu)
        
        self.iota = x[0]
        self.lamb = x[1]*self.effectiveSR
        self.beta = x[2]*self.effectiveSR

        if method == 'Static':
            self.sachs = Static()


    

        self.a2 = np.zeros((self.nsteps,3,3))
        self.a4 = np.zeros((self.nsteps,3,3,3,3))

        for i in range(self.nsteps):
            
            self.a2[i,...] = a2calc(self.n,self.m)
            self.a4[i,...] = a4calc(self.n,self.m)

            if method == 'ImprovedEuler':
                self.n,self.m = self.ImprovedEuler(self.n,self.m,dt)
            elif method == 'ForwardEuler':
                self.n,self.m = self.ForwardEuler(self.n,self.m,dt)
            elif method == 'Static':
                self.n,self.m = self.StaticImprovedEuler(self.n,self.m,dt,self.a2[i,...],self.a4[i,...])
            else:
                print('Method not implemented')
                return

            self.add_delete()
    
            


    
    def weiner(self,n,dt,sigma):
        """
        Add Brownian motion to a set of unit vectors on the surface of a sphere
        by adding normally distributed noise to the x, y, and z coordinates,
        and then re-normalizing the vectors.
        
        vectors: numpy array, initial unit vectors (shape: (n, 3))
        dt: float, time step
        sigma: float, standard deviation of the Gaussian distribution
        """

        noise = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size=(n.shape))
        n2 = noise + n
        norms = np.linalg.norm(n2, axis=0)
        n2  = n2/norms
        return n2 - n


    def add_delete(self):
        # Add and delete particles:
        # If particle has a mass less than tol, delete it
        # For each deleted particle, split the largest particle into two

        # Find particles to delete
        delete = np.where(self.m<self.tol)[0]
        n_to_delete = len(delete)

        # Delete particles
        self.n = np.delete(self.n,delete,axis=1)
        self.m = np.delete(self.m,delete)

        # Find largest n particles
        largest = np.argsort(self.m)[::-1][:n_to_delete]

        # Split largest particles
        self.n = np.concatenate((self.n,self.n[:,largest]),axis=1)
        self.m[largest] /= 2
        self.m = np.concatenate((self.m,self.m[largest]))

        ## Normalise masses
        self.m /= np.mean(self.m)






def a2calc(n,m=1):
    # Calculate 2nd order orientation tensor
    
    A = np.zeros((3,3))

    # ninj = np.einsum('ip,jp->ijp',n,n)
    # for i in range(3):
    #     for j in range(3):
    #         A[i,j] = np.mean(m*ninj[i,j,:])

    A[0,0] = np.mean(m*n[0,:]*n[0,:])
    A[0,1] = np.mean(m*n[0,:]*n[1,:])
    A[0,2] = np.mean(m*n[0,:]*n[2,:])
    A[1,1] = np.mean(m*n[1,:]*n[1,:])
    A[1,2] = np.mean(m*n[1,:]*n[2,:])
    A[2,2] = np.mean(m*n[2,:]*n[2,:])

    A[1,0] = A[0,1]
    A[2,0] = A[0,2]
    A[2,1] = A[1,2]


    return A

def a4calc(n,m=1):
    # Calculate 4th order orientation tensor
    A = np.zeros((3,3,3,3))

    nnnnn = np.einsum('ip,jp,kp,lp->ijklp',n,n,n,n)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A[i,j,k,l] = np.mean(m*nnnnn[i,j,k,l,:])


    return A



def random(npoints):
    # Create n random points on the unit sphere
    n = np.random.randn(npoints,3)
    n /= np.linalg.norm(n,axis=1)[:,None]


    return n.T


def single_max(npoints):
    n = np.zeros((3,npoints))
    n[2,:] = 1
    return n


def isotropic(npoints):
    # Create 100 sets of random points on spheres
    # and calculate the variance of the diagonal of the 2nd order tensor
    # away from 1/3, choose best one
    n = np.random.randn(3,npoints,100)
    n /= np.linalg.norm(n,axis=0)

    a2 = np.zeros((100,3,3))
    std_dev = np.zeros(100)
    for i in range(100):
        a2[i,...] = a2calc(n[:,:,i])
        std_dev[i] = np.std(np.diag(a2[i,...]))

    
    best = np.argmin(std_dev)

    return n[:,:,best]

