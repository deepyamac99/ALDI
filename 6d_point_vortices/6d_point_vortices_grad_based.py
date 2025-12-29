
import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from tqdm import tqdm as tqdm

# Vector field functions with increased eps for stability
def Fx1(x1, x2, x3, y1, y2, y3, Gam2=1, Gam3=-2):
    eps = 0  # Increased from 1e-6 to prevent overflow
    L12 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + eps
    L31 = (x3 - x1) ** 2 + (y3 - y1) ** 2 + eps
    return -(Gam2 * (y1 - y2)) / (2 * np.pi * L12) - (Gam3 * (y1 - y3)) / (2 * np.pi * L31)

def Fx2(x1, x2, x3, y1, y2, y3, Gam1=1, Gam3=-2):
    eps = 0
    L12 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + eps
    L23 = (x2 - x3) ** 2 + (y2 - y3) ** 2 + eps
    return -(Gam1 * (y2 - y1)) / (2 * np.pi * L12) - (Gam3 * (y2 - y3)) / (2 * np.pi * L23)

def Fx3(x1, x2, x3, y1, y2, y3, Gam1=1, Gam2=1):
    eps = 0
    L31 = (x3 - x1) ** 2 + (y3 - y1) ** 2 + eps
    L23 = (x2 - x3) ** 2 + (y2 - y3) ** 2 + eps
    return -(Gam1 * (y3 - y1)) / (2 * np.pi * L31) - (Gam2 * (y3 - y2)) / (2 * np.pi * L23)

def Fy1(x1, x2, x3, y1, y2, y3, Gam2=1, Gam3=-2):
    eps = 0
    L12 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + eps
    L31 = (x3 - x1) ** 2 + (y3 - y1) ** 2 + eps
    return (Gam2 * (x1 - x2)) / (2 * np.pi * L12) + (Gam3 * (x1 - x3)) / (2 * np.pi * L31)

def Fy2(x1, x2, x3, y1, y2, y3, Gam1=1, Gam3=-2):
    eps = 0
    L12 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + eps
    L23 = (x2 - x3) ** 2 + (y2 - y3) ** 2 + eps
    return (Gam1 * (x2 - x1)) / (2 * np.pi * L12) + (Gam3 * (x2 - x3)) / (2 * np.pi * L23)

def Fy3(x1, x2, x3, y1, y2, y3, Gam1=1, Gam2=1):
    eps = 0
    L31 = (x3 - x1) ** 2 + (y3 - y1) ** 2 + eps
    L23 = (x2 - x3) ** 2 + (y2 - y3) ** 2 + eps
    return (Gam1 * (x3 - x1)) / (2 * np.pi * L31) + (Gam2 * (x3 - x2)) / (2 * np.pi * L23)

def yield_X2_X3(X1_ini, Y1_ini, J):
    Gam1, Gam2, Gam3 = 1, 1, -2
    GAM = Gam1 * Gam2 + Gam2 * Gam3 + Gam3 * Gam1
    H = 1
    L = np.exp(-4 * H * np.pi / GAM)

    X1_ini = np.atleast_1d(X1_ini)
    Y1_ini = np.atleast_1d(Y1_ini)
    if X1_ini.size == 1:
        X1_ini = np.full(J, X1_ini.item())
        Y1_ini = np.full(J, Y1_ini.item())

    X2_ini = X1_ini + L
    Y2_ini = Y1_ini
    X12_av = 0.5 * (X1_ini + X2_ini)
    Y12_av = 0.5 * (Y1_ini + Y2_ini)
    V_x = Y1_ini - Y2_ini
    V_y = X2_ini - X1_ini
    norm_V = np.sqrt(V_x**2 + V_y**2)
    norm_V = np.where(norm_V == 0, 1e-10, norm_V)
    X3_ini = X12_av + (np.sqrt(3) * L / (2 * norm_V)) * V_x
    Y3_ini = Y12_av + (np.sqrt(3) * L / (2 * norm_V)) * V_y
    return X2_ini, Y2_ini, X3_ini, Y3_ini

def Euler_alt(X1_ini, X2_ini, X3_ini, Y1_ini, Y2_ini, Y3_ini, u, J, tau, T_f):
    L2 = int(T_f / tau)
    SIG = 2* np.sqrt(tau)

    X1_ini = np.atleast_1d(X1_ini)
    X2_ini = np.atleast_1d(X2_ini)
    X3_ini = np.atleast_1d(X3_ini)
    Y1_ini = np.atleast_1d(Y1_ini)
    Y2_ini = np.atleast_1d(Y2_ini)
    Y3_ini = np.atleast_1d(Y3_ini)

    if X1_ini.size == 1:
        X1_ini = np.full(J, float(X1_ini))
        X2_ini = np.full(J, float(X2_ini))
        X3_ini = np.full(J, float(X3_ini))
        Y1_ini = np.full(J, float(Y1_ini))
        Y2_ini = np.full(J, float(Y2_ini))
        Y3_ini = np.full(J, float(Y3_ini))

    X1 = np.zeros((J, L2 + 1))
    X2 = np.zeros((J, L2 + 1))
    X3 = np.zeros((J, L2 + 1))
    Y1 = np.zeros((J, L2 + 1))
    Y2 = np.zeros((J, L2 + 1))
    Y3 = np.zeros((J, L2 + 1))

    X1[:, 0] = X1_ini
    X2[:, 0] = X2_ini
    X3[:, 0] = X3_ini
    Y1[:, 0] = Y1_ini
    Y2[:, 0] = Y2_ini
    Y3[:, 0] = Y3_ini

    # # Set J_local based on the number of particles in u
    # J_local = u.shape[0] if u.ndim == 2 else 1
    # if u.ndim == 1:
    #     u = u.reshape(1, 3)

    for i in range(L2):
        for j in range(J):
            u_j = u[:,j]
            #print(u_j)
            X1[j, i + 1] = X1[j, i] + tau * Fx1(X1[j, i], X2[j, i], X3[j, i], Y1[j, i], Y2[j, i], Y3[j, i]) + SIG * u_j[0]
            X2[j, i + 1] = X2[j, i] + tau * Fx2(X1[j, i], X2[j, i], X3[j, i], Y1[j, i], Y2[j, i], Y3[j, i]) + SIG * u_j[1]
            X3[j, i + 1] = X3[j, i] + tau * Fx3(X1[j, i], X2[j, i], X3[j, i], Y1[j, i], Y2[j, i], Y3[j, i]) + SIG * u_j[2]
            Y1[j, i + 1] = Y1[j, i] + tau * Fy1(X1[j, i], X2[j, i], X3[j, i], Y1[j, i], Y2[j, i], Y3[j, i]) + SIG * u_j[3]
            Y2[j, i + 1] = Y2[j, i] + tau * Fy2(X1[j, i], X2[j, i], X3[j, i], Y1[j, i], Y2[j, i], Y3[j, i]) + SIG * u_j[4]
            Y3[j, i + 1] = Y3[j, i] + tau * Fy3(X1[j, i], X2[j, i], X3[j, i], Y1[j, i], Y2[j, i], Y3[j, i]) + SIG * u_j[5]
        #print("####################################")

    return X1, X2, X3, Y1, Y2, Y3



def A(x1, x2, x3, y1, y2, y3, Gam1=1, Gam2=1, Gam3=-2):
    GAM = Gam1 * Gam2 + Gam2 * Gam3 + Gam3 * Gam1
    H = 1
    L = np.exp(-4 * H * np.pi / GAM)

    v21 = np.array([x2, y2]) - np.array([x1, y1])
    v31 = np.array([x3, y3]) - np.array([x1, y1])
    v32 = np.array([x3, y3]) - np.array([x2, y2])

    # Add epsilon to norms to prevent division by zero
    norm_v21 = np.linalg.norm(v21)
    norm_v31 = np.linalg.norm(v31)
    norm_v32 = np.linalg.norm(v32)
    norm_v21 = max(norm_v21, 1e-10)
    norm_v31 = max(norm_v31, 1e-10)
    norm_v32 = max(norm_v32, 1e-10)

    cosA = np.dot(v21, v31) / (norm_v21 * norm_v31)
    cosB = -np.dot(v21, v32) / (norm_v21 * norm_v32)

    return abs(cosA - 0.5) + abs(cosB - 0.5) + abs((1/3) * (norm_v21 + norm_v31 + norm_v32) - L)

def A_mean(X1, X2, X3, Y1, Y2, Y3, J, tau, T_f):
    L2 = int(T_f / tau)
    A_X = np.empty((J, L2 + 1))  # Store A values for each particle and time step
    for j in range(J):
        for l in range(L2 + 1):
            A_X[j, l] = A(float(X1[j, l]), float(X2[j, l]), float(X3[j, l]),
                          float(Y1[j, l]), float(Y2[j, l]), float(Y3[j, l]))
    # Average over time steps for each particle
    return np.mean(A_X, axis=1)  # Shape (J,)

def G_atm_alt(u, J, tau=0.2, T_f=3, threshold=0.1):
    J_local = J
    X1_0 = np.zeros(J_local)
    Y1_0 = np.zeros(J_local)
    X2_0, Y2_0, X3_0, Y3_0 = yield_X2_X3(X1_0, Y1_0, J_local)
    X1, X2, X3, Y1, Y2, Y3 = Euler_alt(X1_0, X2_0, X3_0, Y1_0, Y2_0, Y3_0, u, J_local, tau, T_f)
    E = A_mean(X1, X2, X3, Y1, Y2, Y3, J_local, tau, T_f)
    return E - threshold

def G_of_u(u, J):
    return G_atm_alt(u, J, tau=0.02, T_f=0.5, threshold=0.25)







# Smoothed Heaviside surrogate
def G_tilde(u, k=0.001):
    #return np.maximum(0,  G_of_u(u))
    J = u.shape[1]
    x = G_of_u(u,J)
    def psi_delta(x, delta):
        out = np.zeros_like(x)
        mask = x > 0   # flip condition: inside failure
        exponent = 1/delta**2 - 1/x[mask]**2
        exponent = np.clip(exponent, -700, 700)
        out[mask] = np.exp(-exponent)
        return out
    def phi_delta(x, delta):
        num = psi_delta(x, delta)
        denom = num + psi_delta(delta - x, delta)
        return num / denom
    HD = phi_delta(x, delta=k) * (x)   # multiply by -x, not x
    HD=np.nan_to_num(HD, nan=0)
    return HD







# def phi(U,R):
#   Gt =  G_tilde(U)
#   rho =(1/(2*np.pi)**3)*np.exp(-np.sum(U**2,axis=0)/2)
#   #rho = np.clip(rho, 0, 0)   # keep everything ≥ eps_rho
#   rho = np.clip(rho,1e-6,1e+6)
#   return 1/(2*R) * Gt**2 - np.log(rho)

def phi(U, R):
    Gt = G_tilde(U)

    sigma2 = 0.1
    d = U.shape[0]

    rho = (2*np.pi*sigma2)**(-d/2) * np.exp(
        -np.sum(U**2, axis=0) / (2*sigma2)
    )

    return 1/(2*R) * Gt**2 - np.log(rho)



def grad_PHI(U, R, eps=0.001):
    """
    Compute gradient of phi(U,R) w.r.t x and y using finite differences.

    U: array of shape (2, J)
    R: scalar
    eps: small perturbation for finite differences
    Returns grad: array of shape (2, J)
    """
    grad = np.zeros_like(U)  # shape (2, J)

    for i in range(6):  # loop over variables x and y
        U_plus = U.copy()
        U_minus = U.copy()
        U_plus[i,:] += eps
        U_minus[i,:] -= eps

        phi_plus = phi(U_plus, R)   # shape (J,)
        phi_minus = phi(U_minus, R) # shape (J,)

        grad[i,:] = (phi_plus - phi_minus) / (2*eps)

    return grad





# Gradient-based ALDI step
def aldi_gradient_step(U, y, Gamma, dt=0.001, k=1.0, alpha=1.0, eps=1e-6):
    D, N = U.shape
    m = np.mean(U, axis=1, keepdims=True)
    A = U - m

    # Gradient term
    grad_phi = grad_PHI(U, Gamma, k)
    #print(grad_phi)

    # Empirical covariance
    C = (A @ A.T) / (N-1) + eps * np.eye(D)
    C_half = 1/np.sqrt(N-1)*A#np.linalg.cholesky(C)

    # Drift term
    drift = -(C @ grad_phi) + ((D+1)/N) * (U - m)

    # Update per particle
    U_new = U.copy()
    for i in range(N):
        ui = U[:, i:i+1]
        zeta = np.random.randn(N, 1)
        noise = np.sqrt(2 * dt) * (C_half @ zeta)  # Cholesky-based noise
        U_new[:, i:i+1] = ui + dt * drift[:, i:i+1] + noise


    #U_new = np.clip(U_new, 0, 1)
    return U_new

# Run ALDI
def run_aldi_gradient(U0, y, Gamma, n_iter=9000, dt=0.01, k=30.0, alpha=10.0, grad_weight=1.0, data_weight=2.0):
    U = U0.copy()
    hist = {'U': [], 'frac_in_failure': [], 'mean_G': []}
    for it in tqdm(range(n_iter), desc="ALDI iterations"):
        
        frac_in_failure = np.mean(G_of_u(U, U.shape[1]) <= 0)
        mean_G = np.mean(G_of_u(U, U.shape[1]))
        hist['U'].append(U.copy())
        hist['frac_in_failure'].append(frac_in_failure)
        hist['mean_G'].append(mean_G)
        if it % 10 == 0:
            print(f"Iter {it}: Frac in failure domain = {frac_in_failure:.4f} Mean G = {mean_G:.4f}")
        U = aldi_gradient_step(U, y, Gamma, dt=dt, k=k, alpha=alpha)
    return U, hist

if __name__ == "__main__":
  np.random.seed(42)
  for obs in [0.01, 0.001]:
    for sizes in [500]: #,1000, 10000
        
        vv=0.1
        J = sizes
        n_iter = 5000#1_00_000#10_000
        # Initialize near failure boundary
        D=0
        center = np.array([[D], [D],[D], [D],[D], [D]])
        U0 = center + np.random.randn(6, J)*np.sqrt(vv)
        B = 0.001
        y = np.array([[0.0]])
        Gamma = np.array([[obs]])  # Larger Gamma for stability
        DT = 0.001/2
        print(f"Commencing: size {sizes} and R: {obs} with vv: {2} and DT: {DT}")
        U_final, hist = run_aldi_gradient(U0, y, Gamma, n_iter=n_iter, dt=DT, k=B)

  # plot_pf_trajectory(hist, k=B)
        string = f"Final_T05_Long_Vortex_size_{J}_R_{obs}_Delta_{B}_iter_{n_iter}_v_0.1" 
        U_last = np.save(string ,U_final)






