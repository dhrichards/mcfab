import jax.numpy as jnp





def Richards2021(T):
    T[T<-30] = -30
    iotaD = 0.0259733*T + 1.95268104
    iotaS = jnp.zeros_like(T)
    lambtilde = (0.00251776*T + 0.41244777)/jnp.sqrt(2) #correction to agree with SpecCAF
    betatilde = 5*(0.35182521*T + 12.17066493)/jnp.sqrt(2)

    Ecc = jnp.ones_like(T)
    Eca = jnp.ones_like(T)
    power = jnp.ones_like(T)
    

    x = jnp.array([iotaD,iotaS,lambtilde,betatilde,Ecc,Eca,power])
    
    return x.T


def Richards2021Reduced(T,reduce=0.25):
    T[T<-30] = -30
    iotaD = 0.0259733*T + 1.95268104
    iotaS = jnp.zeros_like(T)
    lambtilde = reduce*(0.00251776*T + 0.41244777)/jnp.sqrt(2) #correction to agree with SpecCAF
    betatilde = 5*(0.35182521*T + 12.17066493)/jnp.sqrt(2)

    Ecc = jnp.ones_like(T)
    Eca = jnp.ones_like(T)
    power = jnp.ones_like(T)
    

    x = jnp.array([iotaD,iotaS,lambtilde,betatilde,Ecc,Eca,power])
    
    return x.T

def Elmer(T):
    iotaD = 0.94*jnp.ones_like(T)
    iotaS = 0.6*jnp.ones_like(T)
    lambtilde = jnp.sqrt(2)*2e-3 * jnp.exp(jnp.log(10)*T/10)
    betatilde = jnp.zeros_like(T)
    Ecc = jnp.ones_like(T)
    Eca = 25*jnp.ones_like(T)
    power = jnp.ones_like(T)

    x = jnp.array([iotaD,iotaS,lambtilde,betatilde,Ecc,Eca,power])
    
    return x.T

def Martin2012():
    Ecc = 1.0
    Eca = 10.0
    iotaD = 0.0
    iotaS = Eca/(0.4*Eca + 0.2*Ecc + 0.4)
    power = 1.0

    x = jnp.array([iotaD,iotaS,0.0,0.0,Ecc,Eca,power])

    return x.T


def GrainClass(alpha=1.0,lamb=0.0,beta=0.0,Ecc=1.0,Eca=25.0,power=1.0):
    iotaD = 1-alpha
    iotaS = alpha*Eca/(0.4*Eca + 0.2*Ecc + 0.4)

    x = jnp.array([iotaD,iotaS,lamb,beta,Ecc,Eca,power])

    return x.T