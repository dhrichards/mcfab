import numpy as np

I = np.eye(3)


class Static:
    def __init__(self,gamma=1,beta=0.04,eta=1,alpha=0.04):
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.alpha=alpha

        self.xi1 = (self.gamma + 1)/(4*self.gamma -1) -1/self.beta
        self.xi2 = 1/self.beta -1
        self.xi3 = -(2/3)*((self.gamma+2)/(4*self.gamma-1)-1)


    def __call__(self,A2,A4,gradu,n):

        self.D = 0.5*(gradu + gradu.T)
        self.W = 0.5*(gradu - gradu.T)

        self.F = self.fluidity(A2,A4)
        self.S = np.linalg.tensorsolve(self.F,self.D)

        self.Fstar = self.fluidity_star(n)
        self.Dstar = np.einsum('pijkl,kl->pij',self.Fstar,self.S)


        return self.Dstar
    

    def C(self):
        C = (1-self.alpha)*self.D + self.alpha*self.S/(2*self.eta)
        return C[np.newaxis,:,:]



    def fluidity(self,A2,A4):
    # Fluidity Tensor 
        F = np.zeros_like(A4)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        F[i,j,k,l] = (self.beta/(2*self.eta))*(I[i,k]*I[k,l] \
                        + 2*self.xi1*A4[i,j,k,l] \
                        + self.xi2*(I[i,k]*A2[l,j] + I[j,l]*A2[i,k]) \
                        + self.xi3*A2[k,l]*I[i,j]) \
                        
        return F
    

    def fluidity_star(self,n):
        # Fluidity* tensor
        Fstar = np.zeros((n.shape[0],3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Fstar[:,i,j,k,l] = (self.beta/(2*self.eta))*(I[i,k]*I[k,l] \
                        + 2*self.xi1*n[:,i]*n[:,j]*n[:,k]*n[:,l] \
                        + self.xi2*(I[i,k]*n[:,l]*n[:,j] + I[j,l]*n[:,i]*n[:,k]) \
                        + self.xi3*n[:,k]*n[:,l]*I[i,j]) \
                        

        return Fstar
    


def ind2voigt(i,j):
    if i == 0 and j == 0:
        return 0
    elif i == 1 and j == 1:
        return 1
    elif i == 2 and j == 2:
        return 2
    elif i == 0 and j == 1:
        return 3
    elif i == 0 and j == 2:
        return 4
    elif i == 1 and j == 2:
        return 5
    else:
        return 6 + i + j
    
def voigt2ind(i):
    if i == 0:
        return 0,0
    elif i == 1:
        return 1,1
    elif i == 2:
        return 2,2
    elif i == 3:
        return 0,1
    elif i == 4:
        return 0,2
    elif i == 5:
        return 1,2