import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
from tqdm import tqdm_notebook as tqdm
from scipy.stats import multivariate_normal, norm

# ---------------------------
# Limit state function
# ---------------------------
def G_of_u(u):
    u = np.asarray(u)
    if u.ndim == 1:
        u1, u2 = u[0], u[1]
        print(np.max(u1), np.max(u2))
        v = 0.1*(u1-u2)**2 - (1/math.sqrt(2))*(u1+u2) + 2.5
        return v
    elif u.ndim == 2:
        u1 = u[0, :]
        u2 = u[1, :]
        v = 0.1*(u1-u2)**2 - (1/math.sqrt(2))*(u1+u2) + 2.5
        return v
    else:
        raise ValueError("u must be shape (2,) or (2,N)")

def is_failure(u):
    return np.maximum(0, G_of_u(u))

# ---------------------------
# Full stochastic gradient-free ALDI step
# ---------------------------
def aldi_gradient_free_step(U, y, Gamma, P0_inv, mu0, dt=0.1, eps=1e-12):
    D, N = U.shape
    one_N = np.ones((1, N))
    m = np.mean(U, axis=1, keepdims=True)
    A = U - m @ one_N
    # Forward model
    P = is_failure(U).reshape(1, N)
    p_mean = np.mean(P, axis=1, keepdims=True)

    B = P - p_mean

    # Empirical covariances
    C = (A @ A.T) / (N) #+ eps * np.eye(D)
    C_half = 1/np.sqrt(N-1)*A
    C_uG = (A @ B.T) / (N)
    R_inv = np.linalg.inv(Gamma)


    U_new = U.copy()
    for i in range(N):
        ui = U[:, i:i+1]
        Gi = P[:, i:i+1]

        # Data drift
        data_dir =  (C_uG @ (R_inv @ (Gi - y)))

        # Prior drift
        prior_dir =  C @ (P0_inv @ (ui - mu0))

        # Finite-N correction
        corr = ((D + 1)/N) * (ui - m)

        drift = - data_dir - prior_dir + corr

        # Noise via anomalies
        zeta = np.random.randn(N,1)
        noise =  np.sqrt(2*dt) * (C_half @ zeta)

        # Euler-Maruyama update
        U_new[:, i:i+1] = ui + dt*drift + noise


    return U_new

# ---------------------------
# Run full ALDI
# ---------------------------
def run_aldi(U0, y, Gamma, P0_inv, mu0, n_iter, dt):
    U = U0.copy()
    hist = { 'U': [], 'frac_in_failure': [], 'mean_G': []}
    for it in tqdm(range(n_iter), desc="ALDI iterations"):
      frac_in_failure = np.mean(G_of_u(U) <= 0)
      mean_G = np.mean(G_of_u(U))
      hist['U'].append(U.copy())
      hist['frac_in_failure'].append(frac_in_failure)
      hist['mean_G'].append(mean_G)
      if it % 100 == 0:
          print(f"Iter {it}: Frac in failure domain = {frac_in_failure:.4f}, Mean G = {mean_G:.4f}")
      U = aldi_gradient_free_step(U, y, Gamma,P0_inv, mu0,  dt=dt)
    return U, hist

# ---------------------------
# Plotting functions
# ---------------------------
def plot_ensemble_scatter(hist, iters_to_plot=[0, -1], xlim=(-8,8), ylim=(-8,8)):
    fig, axes = plt.subplots(1, len(iters_to_plot), figsize=(6*len(iters_to_plot),5))
    if len(iters_to_plot) == 1:
        axes = [axes]
    for ax, it in zip(axes, iters_to_plot):
        U = hist["U"][it]
        ax.scatter(U[0, :], U[1, :], alpha=0.5)
        xx, yy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
        zz = G_of_u(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[0], colors='red', linewidths=2)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"ALDI ensemble iter {it}")
        ax.set_xlabel("u1"); ax.set_ylabel("u2")
    plt.tight_layout()
    plt.show()

def plot_pf_trajectory(hist):
    plt.figure(figsize=(20,4))
    plt.plot(hist['pf_est'], marker='o', markersize=2)
    plt.axhline(y=4.21e-3, color='r', linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Pf")
    plt.title("Failure probability over ALDI iterations")
    plt.grid(True)
    plt.show()

obs = np.array([0.1])
sizes = [ 250, 500, 1000]

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
  for ll in obs:
    for i in [10_000]:
      np.random.seed(42)

      # Ensemble size
      J = i


      # Number of ALDI iterations
      n_iter = 10_000


      # Observation and noise
      y = np.array([[0.0]])
      Gamma = np.array([[ll]])


      U0 = np.random.randn(2, J)



      # Add prior for stability
      #P0_inv = inv(np.cov(U0))#np.eye(2)  # Prior covariance inverse
      mu0 = np.zeros((2,1))
      Sigma0 = np.eye(2)
      P0_inv = np.linalg.inv(Sigma0)
      # Initial ensemble (smaller spread for stability)
      U0 += np.array([[-1], [-1]])



      # Run full gradient-free ALDI
      U_final, hist = run_aldi(U0, 0, Gamma, P0_inv, mu0, n_iter=n_iter, dt=0.001)

      # Plot ensemble evolution
      plt.figure(figsize=(10,10))
      plt.scatter(U_final[0,:] , U_final[1,:], s=8, alpha=0.5)

      string = "Ensemble_size_" + str(J) + "_R_" + str(Gamma[0][0]) + "G_free"
      plt.savefig(string+ ".png")
      np.save(string, U_final)

