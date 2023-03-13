import numpy as np
from .static import Static

class solver:
    def __init__(self,npoints,tol=1e-2,inital_condition='isotropic',method='Taylor',integrator='ImprovedEuler'):
        self.npoints = npoints
        self.m = np.ones(npoints)
        self.tol = tol

        if inital_condition == 'isotropic':
            self.n = isotropic(npoints)
        elif inital_condition == 'random':
            self.n = random(npoints)
        elif inital_condition == 'single_max':
            self.n = single_max(npoints)

        self.method = method
        self.integrator = integrator

        self.a2 = a2calc(self.n,self.m)
        self.a4 = a4calc(self.n,self.m)




    def gradu_measures(self,gradu):
        self.gradu = gradu
        self.D = 0.5*(gradu + gradu.T)
        self.W = 0.5*(gradu - gradu.T)
        self.D2 = np.einsum('ij,ji',self.D,self.D)
        self.effectiveSR = np.sqrt(0.5*self.D2)

        


    def v_star(self,n):

        
        Dn = np.einsum('pij,pj->pi',self.Dstar,n)
        Wn = np.einsum('ij,pj->pi',self.W,n)

        Dnnn = np.einsum('pkl,pl,pk,pi->pi',self.Dstar,n,n,n)
        

        v = Wn -self.iota*(Dn - Dnnn)

        return v
    
    
    def V_star(self,n):


        Dnn = np.einsum('pij,pj,pi->p',self.Dstar,n,n)
        

        V = self.W - self.iota*\
            (self.Dstar - np.einsum('p,ij->pij',Dnn,np.eye(3)))
            
       
        return V




    def GBM(self,n,m):


        Dn = np.einsum('ij,pj->pi',self.D,n)

        Dnn = np.einsum('ij,pj,pi->p',self.D,n,n)
        Def = 5*(np.einsum('pi,pi->p',Dn,Dn) - Dnn**2)/self.D2

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

        n = n + 0.5*(k1+k2) + self.weiner(n,dt)
        m = 0.5*(m1+m2)

        n = n/np.linalg.norm(n,axis=1,keepdims=True)
        m /= np.mean(m)

        return n,m

    


    def ForwardEuler(self,n,m,dt):
        # Forward Euler method for SDE
        n += self.v_star(n)*dt + self.weiner(n,dt)
        m = m*(1+self.beta*self.GBM(n,m)*dt)

        n = n/np.linalg.norm(n,axis=1,keepdims=True)
        m /= np.mean(m)

        return n,m
    
    def BackwardEuler(self,n,m,dt):
        # Backward Euler method for SDE
        LHS = np.eye(3) - dt*self.V_star(n)
        RHS = n + dt*self.weiner(n,dt)
        n = np.linalg.solve(LHS,RHS)
        n = n/np.linalg.norm(n,axis=1,keepdims=True)

        m = m*(1+self.beta*self.GBM(n,m)*dt)
        m /= np.mean(m)

        return n,m

    def iterate(self,gradu,dt,x):
        self.gradu_measures(gradu)


        if np.isscalar(x):
            self.params(x)
        else:
            self.iota = x[0]
            self.lamb = x[1]*self.effectiveSR
            self.beta = x[2]*self.effectiveSR


        if self.method == 'Taylor':
            self.Dstar = self.D[np.newaxis,:,:]
        elif self.method == 'Static':
            self.Dstar = self.sachs(self.a2,self.a4,self.gradu,self.n)
        elif self.method == 'C':
            self.sachs(self.a,self.a4,self.gradu,self.n)
            self.Dstar = self.sachs.C()

        if self.integrator == 'ImprovedEuler':
            self.n,self.m = self.ImprovedEuler(self.n,self.m,dt)
        elif self.integrator == 'ForwardEuler':
            self.n,self.m = self.ForwardEuler(self.n,self.m,dt)
        elif self.integrator == 'BackwardEuler':
            self.n,self.m = self.BackwardEuler(self.n,self.m,dt)
        else:
            print('Method not implemented')
            return
    

        self.add_delete()

        self.a2 = a2calc(self.n,self.m)
        self.a4 = a4calc(self.n,self.m)






    def solve_constant(self,gradu,dt,tmax,x,method='Taylor',integrator='ImprovedEuler',**kwargs):
        self.t  = np.arange(0,tmax,dt)
        self.nsteps = len(self.t)

        self.gradu_measures(gradu)
        
        
        if np.isscalar(x):
            self.params(x)
        else:
            self.iota = x[0]
            self.lamb = x[1]*self.effectiveSR
            self.beta = x[2]*self.effectiveSR
       
       
        if method == 'Static' or method == 'C':
            self.sachs = Static(**kwargs)


    

        self.a2 = np.zeros((self.nsteps,3,3))
        self.a4 = np.zeros((self.nsteps,3,3,3,3))

        self.a2[0,...] = a2calc(self.n,self.m)
        self.a4[0,...] = a4calc(self.n,self.m)

        for i in range(1,self.nsteps):
            
            


            if self.method == 'Taylor':
                self.Dstar = self.D[np.newaxis,:,:]
            elif self.method == 'Static':
                self.Dstar = self.sachs(self.a2[i-1,...],self.a4[i-1,...],self.gradu,self.n)
            elif self.method == 'C':
                self.sachs(self.a2[i-1,...],self.a4[i-1,...],self.gradu,self.n)
                self.Dstar = self.sachs.C()

            if self.integrator == 'ImprovedEuler':
                self.n,self.m = self.ImprovedEuler(self.n,self.m,dt)
            elif self.integrator == 'ForwardEuler':
                self.n,self.m = self.ForwardEuler(self.n,self.m,dt)
            elif self.integrator == 'BackwardEuler':
                self.n,self.m = self.BackwardEuler(self.n,self.m,dt)
            else:
                print('Method not implemented')
                return
            

            self.add_delete()

            self.a2[i,...] = a2calc(self.n,self.m)
            self.a4[i,...] = a4calc(self.n,self.m)
        


    def params(self,T):
        self.iota = 0.0259733*T + 1.95268104
        self.lambtilde = (0.00251776*T + 0.41244777)
        self.betatilde = (0.35182521*T + 12.17066493)
        
        self.lamb = self.lambtilde*self.effectiveSR
        self.beta = self.betatilde*self.effectiveSR

    
            


    
    def weiner(self,n,dt):
        """
        Add Brownian motion to a set of unit vectors on the surface of a sphere
        by adding normally distributed noise to the x, y, and z coordinates,
        and then re-normalizing the vectors.
        
        vectors: numpy array, initial unit vectors (shape: (n, 3))
        dt: float, time step
        sigma: float, standard deviation of the Gaussian distribution
        """
        sigma = np.sqrt(2*self.lamb*dt)
        noise = np.random.normal(loc=0, scale=sigma, size=(n.shape))
        n2 = noise + n
        norms = np.linalg.norm(n2, axis=1, keepdims=True)
        n2  = n2/norms
        return n2 - n
    

    def add_brownian_motion(self,n, dt):
        """
        Add Brownian motion to a set of unit vectors on the surface of a sphere.

        less

        vectors: numpy array, initial unit vectors (shape: (n, 3))
        dt: float, time step
        sigma: float, standard deviation of the Gaussian distribution
        """

        # Sample random unit vectors for rotation axes
        random_unit_vectors = random(self.npoints)

        # Compute rotation axes
        rotation_axes = np.cross(n, random_unit_vectors)
        norms = np.linalg.norm(rotation_axes, axis=1)
        rotation_axes /= norms[:, np.newaxis]

        # Sample random rotation angles
        sigma = np.sqrt(2*self.lamb*dt)
        x = np.random.normal(loc=0, scale=sigma, size=self.npoints)
        y = np.random.normal(loc=0, scale=sigma, size=self.npoints)
        rotation_angles = np.sqrt(x**2 + y**2)

        # Correction from "A “Gaussian” for diffusion on the sphere" Ghosh et al. 2012
        rotation_angles = np.sqrt(np.sin(rotation_angles)*rotation_angles)
        
        # Compute rotation matrices
        c, s = np.cos(rotation_angles), np.sin(rotation_angles)
        C = 1 - c
        x, y, z = rotation_axes.T
        R = np.stack([x*x*C + c, x*y*C - z*s, x*z*C + y*s,
                    y*x*C + z*s, y*y*C + c, y*z*C - x*s,
                    z*x*C - y*s, z*y*C + x*s, z*z*C + c], axis=-1).reshape(-1, 3, 3)

        # Rotate vectors
        n2 = np.einsum("pij,pj->pi", R,n)
        n2 = n2/np.linalg.norm(n2,axis=1,keepdims=True)

        return n2 - n


    def add_delete(self):
        # Add and delete particles:
        # If particle has a mass less than tol, delete it
        # For each deleted particle, split the largest particle into two

        # Find particles to delete
        delete = np.where(self.m<self.tol)[0]
        n_to_delete = len(delete)

        # Delete particles
        self.n = np.delete(self.n,delete,axis=0)
        self.m = np.delete(self.m,delete)

        # Find largest n particles
        largest = np.argsort(self.m)[::-1][:n_to_delete]

        # Split largest particles
        self.n = np.concatenate((self.n,self.n[largest,:]),axis=0)
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

    A[0,0] = np.mean(m*n[:,0]*n[:,0])
    A[0,1] = np.mean(m*n[:,0]*n[:,1])
    A[0,2] = np.mean(m*n[:,0]*n[:,2])
    A[1,1] = np.mean(m*n[:,1]*n[:,1])
    A[1,2] = np.mean(m*n[:,1]*n[:,2])
    A[2,2] = np.mean(m*n[:,2]*n[:,2])

    A[1,0] = A[0,1]
    A[2,0] = A[0,2]
    A[2,1] = A[1,2]


    return A

def a4calc(n,m=1):
    # Calculate 4th order orientation tensor
    A = np.zeros((3,3,3,3))

    nnnnn = np.einsum('pi,pj,pk,pl->pijkl',n,n,n,n)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A[i,j,k,l] = np.mean(m*nnnnn[:,i,j,k,l])


    return A



def random(npoints):
    # Create n random points on the unit sphere
    n = np.random.randn(npoints,3)
    n /= np.linalg.norm(n,axis=1, keepdims=True)


    return n


def single_max(npoints):
    n = np.zeros((npoints,3))
    n[:,2] = 1
    return n


def isotropic(npoints):
    # Create 100 sets of random points on spheres
    # and calculate the variance of the diagonal of the 2nd order tensor
    # away from 1/3, choose best one
    n = np.random.randn(npoints,3,100)
    norm = np.linalg.norm(n,axis=1, keepdims=True)
    n /= norm

    a2 = np.zeros((100,3,3))
    std_dev = np.zeros(100)
    for i in range(100):
        a2[i,...] = a2calc(n[:,:,i])
        std_dev[i] = np.std(np.diag(a2[i,...]))

    
    best = np.argmin(std_dev)

    return n[:,:,best]

