import jax.numpy as jnp
import jax

def Selm(D,A2,A4,gamma,beta,eta):
    # Stress tensor
     
    F = fluidity(A2,A4,gamma,beta,eta)
    S = jnp.linalg.tensorsolve(F,D)

    return S
    

def xi_calc(gamma,beta,eta):
        xi0 = beta/(2*eta)
        xi1 = (gamma + 1)/(4*gamma -1) -1/beta
        xi2 = 1/beta -1
        xi3 = -(2/3)*((gamma+2)/(4*gamma-1)-1)

        return jnp.array([xi0,xi1,xi2,xi3])


def Celm(D,A2,A4,alpha,gamma=1,beta=0.04,eta=1,ks=10):
    # C based on Gillet-Chaulet 2006
    S = Selm(D,A2,A4,gamma,beta,eta)

    C = (1-alpha)*D + alpha*ks*S/(2*eta)
    return C


def fluidity(A2,A4,gamma,beta,eta):
    # Fluidity Tensor 
    I = jnp.eye(3)
    xi = xi_calc(gamma,beta,eta)

    F = xi[0]*(jnp.einsum('ik,jl->ijkl',I,I) \
        + 2*xi[1]*A4 \
        + xi[2]*(jnp.einsum('ik,lj->ijkl',I,A2) + jnp.einsum('ik,jl->ijkl',A2,I)) \
        + xi[3]*jnp.einsum('kl,ij->ijkl',A2,I))
    return F
    

def da2(a2,a4,gradu,x):
    
    I = jnp.eye(3)

    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)

    C = Celm(D,a2,a4,x[3],ks=x[4])

    normC = jnp.sqrt(jnp.einsum('ij,ji->',C,C))


    da2 = jnp.einsum('ik,kj->ij',W,a2) - jnp.einsum('ik,kj->ij',a2,W)\
        -x[0]*(jnp.einsum('ik,kj->ij',C,a2) + jnp.einsum('ik,kj->ij',a2,C)\
                -2*jnp.einsum('ijkl,kl->ij',a4,C))\
        + x[1]*(I - 3*a2)*normC
    return da2


def iterate(fabric,p):
    a2,a4 = fabric
    gradu,x,dt = p

    I = jnp.eye(3)

    D = 0.5*(gradu + gradu.T)
    W = 0.5*(gradu - gradu.T)

    C = Celm(D,a2,a4,x[3],ks=x[4])

    normC = jnp.sqrt(jnp.einsum('ij,ji->',C,C))


    L = jnp.einsum('ik,jl->ijkl',W,I) - jnp.einsum('ik,lj->ijkl',I,W) \
        - x[0]*(jnp.einsum('ik,jl->ijkl',C,I) + jnp.einsum('ik,lj->ijkl',I,C))
    
    LHS = jnp.einsum('ik,jl->ijkl',I,I) - dt*L

    RHS = a2 +dt*(-x[0]*2*jnp.einsum('ijkl,kl->ij',a4,C)\
          + x[1]*(I - 3*a2)*normC)
    
    a2 = jnp.linalg.tensorsolve(LHS,RHS)
    a4 = IBOF_closure(a2)

    return (a2,a4),fabric


def iterate_rk4(fabric,p):
    # Iterate using a 4th order Runge-Kutta method
    a2,a4 = fabric
    gradu,x,dt = p

    k1 = dt*da2(a2,a4,gradu,x)
    k2 = dt*da2(a2+0.5*k1,a4,gradu,x)
    k3 = dt*da2(a2+0.5*k2,a4,gradu,x)
    k4 = dt*da2(a2+k3,a4,gradu,x)

    a2 = a2 + (k1 + 2*k2 + 2*k3 + k4)/6
    a4 = IBOF_closure(a2)

    return (a2,a4),fabric




def time(dt,tmax):
    t  = jnp.arange(0,tmax,dt)
    nsteps = len(t)
    return t,nsteps

def tile_arrays(gradu,dt,tmax,x):

    t,nsteps = time(dt,tmax)
    # Tile arrays to be nsteps long
    gradu_tile = jnp.tile(gradu,(nsteps,1,1))
    dt_tile = jnp.tile(dt,(nsteps,))
    x_tile = jnp.tile(x,(nsteps,1))

    return gradu_tile,dt_tile,x_tile


def solve(gradu,dt,x):
    """Solve the SDE using the lax scan function
    npoints is the number of particles
    gradu is the velocity gradient (nsteps,3,3))
    dt is the time step (nsteps,)
    x is the non-dimensional parameter vector (nsteps,3)"""

    a2 = jnp.eye(3)/3
    a4 = IBOF_closure(a2)

    fabric_0 = (a2,a4)


    final,fabric = jax.lax.scan(iterate_rk4,fabric_0,(gradu,x,dt))

    return fabric


def params(T):
    iota = jnp.ones_like(T)
    lambtilde = 2e-3 * jnp.exp(jnp.log(10)*T/10)
    betatilde = jnp.zeros_like(T)
    alpha = 0.06*jnp.ones_like(T)
    ks = 10*jnp.ones_like(T)

    x = jnp.array([iota,lambtilde,betatilde,alpha,ks])
    
    return x.T

def IBOF_closure(a):
    """Generate IBOF closure.
    Parameters
    ----------
    a : 3x3 numpy array
        Second order fiber orientation tensor.
    Returns
    -------
    3x3x3x3 numpy array
        Fourth order fiber orientation tensor.
    References
    ----------
    .. [1] Du Hwan Chung and Tai Hun Kwon,
       'Invariant-based optimal fitting closure approximation for the numerical
       prediction of flow-induced fiber orientation',
       Journal of Rheology 46(1):169-194,
       https://doi.org/10.1122/1.1423312
    """

    # second invariant
    II = (
        a[..., 0, 0] * a[..., 1, 1]
        + a[..., 1, 1] * a[..., 2, 2]
        + a[..., 0, 0] * a[..., 2, 2]
        - a[..., 0, 1] * a[..., 1, 0]
        - a[..., 1, 2] * a[..., 2, 1]
        - a[..., 0, 2] * a[..., 2, 0]
    )

    # third invariant
    III = jnp.linalg.det(a)

    # coefficients from Chung & Kwon paper
    C1 = jnp.zeros((1, 21))

    C2 = jnp.zeros((1, 21))

    C3 = jnp.array(
        [
            [
                0.24940908165786e2,
                -0.435101153160329e3,
                0.372389335663877e4,
                0.703443657916476e4,
                0.823995187366106e6,
                -0.133931929894245e6,
                0.880683515327916e6,
                -0.991630690741981e7,
                -0.159392396237307e5,
                0.800970026849796e7,
                -0.237010458689252e7,
                0.379010599355267e8,
                -0.337010820273821e8,
                0.322219416256417e5,
                -0.257258805870567e9,
                0.214419090344474e7,
                -0.449275591851490e8,
                -0.213133920223355e8,
                0.157076702372204e10,
                -0.232153488525298e5,
                -0.395769398304473e10,
            ]
        ]
    )

    C4 = jnp.array(
        [
            [
                -0.497217790110754e0,
                0.234980797511405e2,
                -0.391044251397838e3,
                0.153965820593506e3,
                0.152772950743819e6,
                -0.213755248785646e4,
                -0.400138947092812e4,
                -0.185949305922308e7,
                0.296004865275814e4,
                0.247717810054366e7,
                0.101013983339062e6,
                0.732341494213578e7,
                -0.147919027644202e8,
                -0.104092072189767e5,
                -0.635149929624336e8,
                -0.247435106210237e6,
                -0.902980378929272e7,
                0.724969796807399e7,
                0.487093452892595e9,
                0.138088690964946e5,
                -0.160162178614234e10,
            ]
        ]
    )

    C5 = jnp.zeros((1, 21))

    C6 = jnp.array(
        [
            [
                0.234146291570999e2,
                -0.412048043372534e3,
                0.319553200392089e4,
                0.573259594331015e4,
                -0.485212803064813e5,
                -0.605006113515592e5,
                -0.477173740017567e5,
                0.599066486689836e7,
                -0.110656935176569e5,
                -0.460543580680696e8,
                0.203042960322874e7,
                -0.556606156734835e8,
                0.567424911007837e9,
                0.128967058686204e5,
                -0.152752854956514e10,
                -0.499321746092534e7,
                0.132124828143333e9,
                -0.162359994620983e10,
                0.792526849882218e10,
                0.466767581292985e4,
                -0.128050778279459e11,
            ]
        ]
    )

    # build matrix of coefficients by stacking vectors
    C = jnp.vstack((C1, C2, C3, C4, C5, C6))

    # compute parameters as fith order polynom based on invariants
    beta3 = (
        C[2, 0]
        + C[2, 1] * II
        + C[2, 2] * II**2
        + C[2, 3] * III
        + C[2, 4] * III**2
        + C[2, 5] * II * III
        + C[2, 6] * II**2 * III
        + C[2, 7] * II * III**2
        + C[2, 8] * II**3
        + C[2, 9] * III**3
        + C[2, 10] * II**3 * III
        + C[2, 11] * II**2 * III**2
        + C[2, 12] * II * III**3
        + C[2, 13] * II**4
        + C[2, 14] * III**4
        + C[2, 15] * II**4 * III
        + C[2, 16] * II**3 * III**2
        + C[2, 17] * II**2 * III**3
        + C[2, 18] * II * III**4
        + C[2, 19] * II**5
        + C[2, 20] * III**5
    )

    beta4 = (
        C[3, 0]
        + C[3, 1] * II
        + C[3, 2] * II**2
        + C[3, 3] * III
        + C[3, 4] * III**2
        + C[3, 5] * II * III
        + C[3, 6] * II**2 * III
        + C[3, 7] * II * III**2
        + C[3, 8] * II**3
        + C[3, 9] * III**3
        + C[3, 10] * II**3 * III
        + C[3, 11] * II**2 * III**2
        + C[3, 12] * II * III**3
        + C[3, 13] * II**4
        + C[3, 14] * III**4
        + C[3, 15] * II**4 * III
        + C[3, 16] * II**3 * III**2
        + C[3, 17] * II**2 * III**3
        + C[3, 18] * II * III**4
        + C[3, 19] * II**5
        + C[3, 20] * III**5
    )

    beta6 = (
        C[5, 0]
        + C[5, 1] * II
        + C[5, 2] * II**2
        + C[5, 3] * III
        + C[5, 4] * III**2
        + C[5, 5] * II * III
        + C[5, 6] * II**2 * III
        + C[5, 7] * II * III**2
        + C[5, 8] * II**3
        + C[5, 9] * III**3
        + C[5, 10] * II**3 * III
        + C[5, 11] * II**2 * III**2
        + C[5, 12] * II * III**3
        + C[5, 13] * II**4
        + C[5, 14] * III**4
        + C[5, 15] * II**4 * III
        + C[5, 16] * II**3 * III**2
        + C[5, 17] * II**2 * III**3
        + C[5, 18] * II * III**4
        + C[5, 19] * II**5
        + C[5, 20] * III**5
    )

    beta1 = (
        3.0
        / 5.0
        * (
            -1.0 / 7.0
            + 1.0
            / 5.0
            * beta3
            * (1.0 / 7.0 + 4.0 / 7.0 * II + 8.0 / 3.0 * III)
            - beta4 * (1.0 / 5.0 - 8.0 / 15.0 * II - 14.0 / 15.0 * III)
            - beta6
            * (
                1.0 / 35.0
                - 24.0 / 105.0 * III
                - 4.0 / 35.0 * II
                + 16.0 / 15.0 * II * III
                + 8.0 / 35.0 * II**2
            )
        )
    )

    beta2 = (
        6.0
        / 7.0
        * (
            1.0
            - 1.0 / 5.0 * beta3 * (1.0 + 4.0 * II)
            + 7.0 / 5.0 * beta4 * (1.0 / 6.0 - II)
            - beta6
            * (
                -1.0 / 5.0
                + 2.0 / 3.0 * III
                + 4.0 / 5.0 * II
                - 8.0 / 5.0 * II**2
            )
        )
    )

    beta5 = (
        -4.0 / 5.0 * beta3
        - 7.0 / 5.0 * beta4
        - 6.0 / 5.0 * beta6 * (1.0 - 4.0 / 3.0 * II)
    )

    # second order identy matrix
    delta = jnp.eye(3)

    # generate fourth order tensor with parameters and tensor algebra
    return (
        symm(jnp.einsum("..., ij,kl->...ijkl", beta1, delta, delta))
        + symm(jnp.einsum("..., ij, ...kl-> ...ijkl", beta2, delta, a))
        + symm(jnp.einsum("..., ...ij, ...kl -> ...ijkl", beta3, a, a))
        + symm(
            jnp.einsum("..., ij, ...km, ...ml -> ...ijkl", beta4, delta, a, a)
        )
        + symm(
            jnp.einsum("..., ...ij, ...km, ...ml -> ...ijkl", beta5, a, a, a)
        )
        + symm(
            jnp.einsum(
                "..., ...im, ...mj, ...kn, ...nl -> ...ijkl", beta6, a, a, a, a
            )
        )
    )



def symm(A):
    """Symmetrize the fourth order tensor.
    This function computes the symmetric part of a fourth order Tensor A
    and returns a symmetric fourth order tensor S.
    """
    permutations = [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
        (0, 2, 1, 3),
        (2, 0, 1, 3),
        (0, 2, 3, 1),
        (2, 0, 3, 1),
        (1, 3, 0, 2),
        (3, 1, 0, 2),
        (1, 3, 2, 0),
        (3, 1, 2, 0),
        (0, 3, 1, 2),
        (3, 0, 1, 2),
        (0, 3, 2, 1),
        (3, 0, 2, 1),
        (1, 2, 0, 3),
        (2, 1, 0, 3),
        (1, 2, 3, 0),
        (2, 1, 3, 0),
    ]

    S = sum(
        jnp.einsum(A, [Ellipsis] + list(perm))
        for perm in permutations
    ) / 24.0

    return S