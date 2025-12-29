

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.stats import multivariate_normal
from typing import Tuple, Callable


from tqdm import tqdm_notebook as tqdm

plt.rcParams.update({
    # --- Figure layout ---
    'figure.dpi': 300,                # high-resolution output
    'savefig.dpi': 300,               # higher resolution when saving

    # --- Lines and markers ---
    'lines.linewidth': 4.0,           # thicker lines
    'lines.markersize': 6,            # medium marker size
    'lines.markeredgewidth': 0.8,     # marker edge thickness

    # --- Fonts and labels ---
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,

    # --- Grid and ticks ---
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.linewidth': 1.4,            # thicker border around plots
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,


    # --- Legends ---
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
})




def plotter(xx,yy,zz):
  # Create one figure with two subplots (1 row, 2 columns)
  fig = plt.figure(figsize=(14, 6))

  # --- 2D Contour plot ---
  ax1 = fig.add_subplot(1, 2, 1)
  cont = ax1.contourf(xx, yy, zz, levels=200, cmap='viridis')

  tol = 1e-6

  ax1.contour(xx, yy, zz, levels=[-tol,+tol], colors='red', linewidths=10)
  fig.colorbar(cont, ax=ax1, shrink=0.7)

  # --- 3D Surface plot ---
  ax2 = fig.add_subplot(1, 2, 2, projection='3d')
  surf = ax2.plot_surface(xx, yy, zz, cmap='viridis',
                          edgecolor='none', rstride=1, cstride=1,
                          vmin=np.min(zz), vmax=np.max(zz))
  fig.colorbar(surf, ax=ax2, shrink=0.7)

  plt.tight_layout()
  plt.show()
  return None

  # Plotting
def plot_ensemble_scatter(hist, iters_to_plot=[0, -1], xlim=(-10,10), ylim=(-10,10), k=30.0):
    fig, axes = plt.subplots(1, len(iters_to_plot), figsize=(6*len(iters_to_plot),5))
    if len(iters_to_plot) == 1:
        axes = [axes]
    for ax, it in zip(axes, iters_to_plot):
        U = hist["U"][it]
        ax.scatter(U[0, :], U[1, :], alpha=0.5)
        xx, yy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
        zz = G_of_u(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[0], colors='red', linewidths=4)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"ALDI ensemble iter {it}, Delta={k}")
        ax.set_xlabel("u1"); ax.set_ylabel("u2")
    plt.tight_layout()
    plt.show()





def G_of_u(u):
    u = np.asarray(u)
    if u.ndim == 1:
        u1, u2 = u[0], u[1]
        return 0.1*(u1-u2)**2 - (1/np.sqrt(2))*(u1+u2) + 2.5
    elif u.ndim == 2:
        u1, u2 = u[0,:], u[1,:]
        return 0.1*(u1-u2)**2 - (1/np.sqrt(2))*(u1+u2) + 2.5
    else:
        raise ValueError("u must be shape (2,) or (2,N)")

# Smoothed Heaviside surrogate
def G_tilde(u, k=1.0):
    #return np.maximum(0,  G_of_u(u))
    x = G_of_u(u)
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

# Example data
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
xx, yy = np.meshgrid(x, y)

U = np.vstack([xx.ravel(), yy.ravel()])
zz = G_of_u(U)
zz = zz.reshape(xx.shape)

plotter(xx,yy,zz)
print(f"Minimum value: {np.min(zz)}")
print(f"Maximum value: {np.max(zz)}")



"""#Modified limit state function:
For consistency w.r.t bayesian setting the limit state function is modified as,
$\tilde{G(u)}:= max(0, G(u))$.
"""

def G_tilde(u, k=1.0):
  return np.maximum(0,  G_of_u(u))

# Example data
x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
xx, yy = np.meshgrid(x, y)

U = np.vstack([xx.ravel(), yy.ravel()])
zz = G_tilde(U)
zz = zz.reshape(xx.shape)

plotter(xx,yy,zz)
print(f"Minimum value: {np.min(zz)}")
print(f"Maximum value: {np.max(zz)}")



def G_tilde(u, k=0.01):
    #return np.maximum(0,  G_of_u(u))
    x = G_of_u(u)
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



def rho_gen(x, mu, Sigma):
    d = len(mu)
    inv_Sigma = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_Sigma)
    diff = x - mu[:, None]
    exponent = -0.5 * np.einsum('ij,ik,jk->k', inv_Sigma, diff, diff)
    return norm_const * np.exp(exponent)

# --- Fixed Gaussian parameters ---


def phi(U,R, k):
  mu = np.array([0, 0])
  Sigma = np.array([[1,0],
                  [0, 1]])
  Gt =  G_tilde(U, k)
  rho = rho_gen(U, mu, Sigma) #(1/(2*np.pi*0.1))*np.exp((-(U[0]-3)**2 -(U[1]-3)**2 )/)
  #rho = np.clip(rho,1e-6,1e+6)
  return 1/(2*R) * Gt**2 - np.log(rho)


# # Example grid
x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
xx, yy = np.meshgrid(x, y)
U = np.vstack([xx.ravel(), yy.ravel()])

# # Evaluate phi
# R = 1
# zz = phi(U, R).reshape(xx.shape)
# plotter(xx,yy,zz)
# print(f"Minimum value: {np.min(zz)}")
# print(f"Maximum value: {np.max(zz)}")

def grad_PHI(U, R, k, eps=0.001):
    """
    Compute gradient of phi(U,R) w.r.t x and y using finite differences.

    U: array of shape (2, J)
    R: scalar
    eps: small perturbation for finite differences
    Returns grad: array of shape (2, J)
    """
    grad = np.zeros_like(U)  # shape (2, J)

    for i in range(2):  # loop over variables x and y
        U_plus = U.copy()
        U_minus = U.copy()
        U_plus[i,:] += eps
        U_minus[i,:] -= eps

        phi_plus = phi(U_plus, R, k)   # shape (J,)
        phi_minus = phi(U_minus, R, k) # shape (J,)

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

    return U_new




# Run ALDI
def run_aldi_gradient(U0, y, Gamma, n_iter=9000, dt=0.01, k=30.0, alpha=10.0, grad_weight=1.0, data_weight=2.0):
    U = U0.copy()
    hist = {'U': [], 'frac_in_failure': [], 'mean_G': []}
    for it in tqdm(range(n_iter), desc="ALDI iterations"):
        frac_in_failure = np.mean(G_of_u(U) <= 0)
        mean_G = np.mean(G_of_u(U))
        hist['U'].append(U.copy())
        hist['frac_in_failure'].append(frac_in_failure)
        hist['mean_G'].append(mean_G)
        if it % 100 == 0:
            print(f"Iter {it}: Frac in failure domain = {frac_in_failure:.4f}, Mean G = {mean_G:.4f}")
        U = aldi_gradient_step(U, y, Gamma, dt=dt, k=k, alpha=alpha)
    return U, hist

obs = np.array([0.00001,  0.0001,  1])#np.array([  0.001, 0.01, 0.1]) #0.01, 0.1,
sizes = [1000] #100, 1000, 10000

# Main
if __name__ == "__main__":
  for i in sizes:
    for ll in obs:
      np.random.seed(42)
      J = i
      nvar = 1
      n_iter =  10000
      # Initialize near failure boundary
      center = np.array([[-1], [-1]])
      U0 = center + np.random.randn(2, J)
      B = ll
      y = np.array([[0.0]])
      Gamma = np.array([[0.01]])  # Larger Gamma for stability
      DT = 0.001#0.001
      U_final, hist = run_aldi_gradient(U0, y, Gamma, n_iter=n_iter, dt=DT, k=B)
      print(U_final.shape)
      plt.figure(figsize=(10,10))
      plt.contour(xx,yy,zz, levels=[0], colors="r")
      plt.scatter(U_final[0,:], U_final[1,:], s=10, alpha=0.5)

      plt.show()
      
      string = "Delta_variation" + str(J) + "_R_" + str(Gamma[0][0]) + "_delta_" + str(B)
      plt.savefig(string + ".png")

      np.save(string, hist["U"])

      print("Ensemble_size_" + str(J) + "_R_" + str(Gamma[0][0]) + "_delta_" + str(B) + "_completed")
























"""Below we print the mean of the entire Estimated failure probability for Delta=0.1."""



















