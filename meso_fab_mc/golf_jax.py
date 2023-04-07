import jax.numpy as jnp
import os
import numpy as np
import jax

def GolfStress(a2,D,fabric_grid):

    a2 = 0.5 * (a2 + a2.T)

    eigvals, eigvecs = jnp.linalg.eigh(a2)

    eta6 = grid_viscosities(fabric_grid, eigvals)
    
    

    S = jnp.zeros((3, 3))
    for r in range(3):
        Mr = jnp.outer(eigvecs[:, r], eigvecs[:, r])

        S += 0.5*((eta6[r])*tr(Mr.dot(D)) * dev(Mr)\
              + (eta6[r+3]) * dev(D.dot(Mr) + Mr.dot(D)))
        
    return S



def tr(X):
    """
    Calculate the trace of a tensor
    """
    return jnp.sum(jnp.diag(X))


def dev(X):
    """
    Calculate the deviatoric part of a tensor
    """
    return X - jnp.mean(jnp.diag(jnp.diag(X))) * jnp.eye(3)

def r2ro(A):

    ai, EigenVec = jnp.linalg.eig(A)

    # Ensure a right-handed orthonormal basis
    # EigenVec[:, 2] = np.cross(EigenVec[:, 0], EigenVec[:, 1])

    # # Normalize
    # EigenVec[:, 2] /= np.linalg.norm(EigenVec[:, 2])

    Euler = jnp.zeros(3)
    Euler = Euler.at[1].set(jnp.arccos(EigenVec[2, 2]))

    if abs(Euler[1]) > jnp.eps:
        # 3D Euler angles
        Euler = Euler.at[0].set(jnp.arctan2(EigenVec[0, 2], -EigenVec[1, 2]))
        Euler = Euler.at[2].set(jnp.arctan2(EigenVec[2, 0], EigenVec[2, 1]))
    else:
        # Only one rotation of angle phi
        Euler = Euler.at[2].set(0.0)
        Euler = Euler.at[0].set(jnp.arctan2(EigenVec[1, 0], EigenVec[0, 0]))

    return ai, Euler, EigenVec

def load_viscosity_data(beta=0.04, gamma=1.0, n=1.0, model='V'):
        # Convert beta, gamma, and n to the corresponding strings in the file name
    beta_str = f"{int(beta * 10000):04d}"
    gamma_str = f"{int(gamma * 100 // 10):02d}{int(gamma * 100 % 10):01d}"
    n_str = f"{int(n):01d}{int((n * 10) % 10):01d}"

    # Build the file name
    file_name = f"{beta_str}{gamma_str}{n_str}.{model}a"

    # Set the folder path
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    #file_path = file_name
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    rows = 813
    cols = 6

    # Initialize an empty array
    fabric_grid = np.zeros((rows * cols))


    with open(file_path, "r") as f:
        for i in range(rows):
            line = f.readline()
            for j in range(cols):
                # Extract the value from the line based on fixed width
                value = float(line[j * 14:(j + 1) * 14])
                fabric_grid[i * cols + j] = value


    return fabric_grid



def grid_viscosities(etaI,eigvals):
    # Grid data
    kmin=0.002
    Ndiv = 30
    Ntot = 813
    NetaI = 4878
    Nk2 = jnp.array([
    -1, 46, 93, 139, 183, 226, 267, 307, 345, 382,
    417, 451, 483, 514, 543, 571, 597, 622, 645, 667,
    687, 706, 723, 739, 753, 766, 777, 787, 795, 802,
    807, 811
    ])
    UnTier = 1 / 3
    Delta = UnTier / Ndiv
    
    #convert etai to jax array
    etaI = jnp.array(etaI)

    ai0_sorted_indices = jnp.argsort(eigvals)
    ai = jnp.sort(eigvals)
    ordre = ai0_sorted_indices + 1

    a1 = ai[0]
    a2 = ai[1]

    ik1 = ((a1 + Delta) / Delta) + 1
    ik2 = ((a2 + Delta) / Delta) + 1

    ik1 = jnp.array(ik1,int)
    ik2 = jnp.array(ik2,int)

    # if ((ik1 + 2 * ik2 - 3 * (Ndiv + 1)) >= 0):
    #     if ((ik1 != 2) and ((ik1 + 2 * ik2 - 3 * (Ndiv + 1)) != 0) and (abs((a1 - Delta * (ik1 - 1)) / a1) > 1.0e-5)):
    #         ik1 = ik1 - 1
    #     ik2 = ik2 - 1
    # Jaxified version of the above
    def update_ik1(ik1):
        return ik1 - 1

    def update_ik2(ik2):
        return ik2 - 1

    def no_update(x):
        return x
    
    # First condition
    predicate1 = (ik1 + 2 * ik2 - 3 * (Ndiv + 1)) >= 0

    # Second condition
    predicate2 = (ik1 != 2) & ((ik1 + 2 * ik2 - 3 * (Ndiv + 1)) != 0) & (jnp.abs((a1 - Delta * (ik1 - 1)) / a1) > 1.0e-5)

    # Update ik1 if both conditions are true
    ik1 = jax.lax.cond(predicate1 & predicate2, update_ik1, no_update, ik1)

    # Update ik2 if the first condition is true
    ik2 = jax.lax.cond(predicate1, update_ik2, no_update, ik2)
    

    #Jax expression for if ik1==1 then ik1=2
    ik1 = jax.lax.cond(ik1 == 1, lambda _: 2, lambda _: ik1, None)
    

    N4 = Nk2[ik1 - 1 -1] + ik2 - ik1 + 3
    N5 = Nk2[ik1 -1] + ik2 - ik1 + 2
    N6 = Nk2[ik1 + 1 -1] + ik2 - ik1 + 1

    # a1i = jnp.array([Delta * (ik1 - 3 + i) for i in range(1, 4)])
    # a2i = jnp.array([Delta * (ik2 - 3 + i) for i in range(1, 4)])
    i = jnp.arange(1, 4)
    a1i = Delta * (ik1 - 3 + i)
    a2i = Delta * (ik2 - 3 + i)

    etaN = jnp.zeros(9)
    eta6 = jnp.zeros(6)


    for n in range(1, 7):
        
        etaN = etaN.at[0 + (i - 1) * 3].set(etaI[6 * (N4 - 3 + i) + n - 1])
        etaN = etaN.at[1 + (i - 1) * 3].set(etaI[6 * (N5 - 3 + i) + n - 1])
        etaN = etaN.at[2 + (i - 1) * 3].set(etaI[6 * (N6 - 3 + i) + n - 1])

        eta6 = eta6.at[n - 1].set(InterQ9(a1, a2, a1i, a2i, etaN))

    # def eta6_loop_body(n, eta6):
    #     etaN = jnp.zeros(9)
    #     etaN = etaN.at[0 + (i - 1) * 3].set(etaI[tuple(6 * (N4 - 3 + i) + n - 1)])
    #     etaN = etaN.at[1 + (i - 1) * 3].set(etaI[tuple(6 * (N5 - 3 + i) + n - 1)])
    #     etaN = etaN.at[2 + (i - 1) * 3].set(etaI[tuple(6 * (N6 - 3 + i) + n - 1)])
    #     eta6 = eta6.at[n - 1].set(InterQ9(a1, a2, a1i, a2i, etaN))
    #     return eta6

    # eta6 = jax.lax.fori_loop(1, 7, eta6_loop_body, eta6)

    eta6sorted = jnp.zeros(6)


    eta6sorted = eta6sorted.at[ordre-1].set(eta6[0:3])
    eta6sorted = eta6sorted.at[ordre+3-1].set(eta6[3:]) 

    return eta6sorted


def InterP(t, x, Q):
    d12 = x[1] - x[0]
    d23 = x[2] - x[1]
    Ip = Q[0] * (x[1] - t) * (x[2] - t) / ((d12 + d23) * d12)
    Ip += Q[1] * (t - x[0]) * (x[2] - t) / (d12 * d23)
    Ip += Q[2] * (t - x[0]) * (t - x[1]) / ((d12 + d23) * d23)

    return Ip

# InterQ9 function
def InterQ9(x, y, xi, yi, Q):
    a = jnp.array([InterP(x, xi, Q[0:3]), InterP(x, xi, Q[3:6]), InterP(x, xi, Q[6:9])])
    Ip = InterP(y, yi, a)

    return Ip