from re import U
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
from tqdm import tqdm as tqdm
from scipy.stats import multivariate_normal, norm

# ---------------------------
# Limit state function
# ---------------------------
def G_of_u(u, t_grid=None):

    #############Ignore##############
    if t_grid is None:
      raise ValueError("t_grid must be provided. It is supposed to be a numpy array e.g., np.linspace(0,final_time, Number_of_points)")
    else:
      pass
    #################################

    alpha = 1
    beta = 1
    mu=1
    u = np.asarray(u)  # Shape: (2, N)
    u1 = u[0, :]  # Shape: (N,)
    u2 = u[1, :]  # Shape: (N,)

    # Compute trajectory over t_grid (broadcasting)
    x = u1[None, :] * np.exp(-alpha * t_grid[:, None])  # Shape: (200, N)
    y = u2[None, :] * np.exp(beta * t_grid[:, None])    # Shape: (200, N)

    # Compute r^2 = x^2 + y^2 for each timestep
    r2 = x**2 + y**2  # Shape: (200, N)


    # Mean r^2 over time for each particle
    mean_r2 = np.mean(r2, axis=0)  # Shape: (N,)

    # Return G = 0.5 - mean_r2
    return mean_r2-0.5  # Shape: (N,)


def is_failure(u, t_grid):
    return np.maximum(0, G_of_u(u,t_grid))

# ---------------------------
# Full stochastic gradient-free ALDI step
# ---------------------------
def aldi_gradient_free_step(U, y, t_grid, Gamma, P0_inv, mu0, dt=0.1, eps=1e-12):
    D, N = U.shape
    one_N = np.ones((1, N))
    m = np.mean(U, axis=1, keepdims=True)
    A = U - m @ one_N
    # Forward model
    P = is_failure(U, t_grid).reshape(1, N)
    p_mean = np.mean(P, axis=1, keepdims=True)

    B = P - p_mean

    # Empirical covariances
    C = (A @ A.T) / (N) + eps * np.eye(D)
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
def run_aldi(U0, y, t_grid, Gamma, P0_inv, mu0, n_iter, dt):
    U = U0.copy()
    hist = { 'U': [], 'frac_in_failure': [], 'mean_G': []}
    for it in tqdm(range(n_iter), desc="ALDI iterations"):
      frac_in_failure = np.mean(G_of_u(U, t_grid) <= 0)
      mean_G = np.mean(G_of_u(U, t_grid))
      hist['U'].append(U.copy())
      hist['frac_in_failure'].append(frac_in_failure)
      hist['mean_G'].append(mean_G)
      if it % 100 == 0:
          print(f"Iter {it}: Frac in failure domain = {frac_in_failure:.4f}, Mean G = {mean_G:.4f}")
      U = aldi_gradient_free_step(U, y,t_grid, Gamma,P0_inv, mu0,  dt=dt)

    return U, hist

# ---------------------------
# Plotting functions
# ---------------------------
def plot_ensemble_scatter(hist,t_grid, iters_to_plot=[0, -1], xlim=(-2,2), ylim=(-2,2)):
    fig, axes = plt.subplots(1, len(iters_to_plot), figsize=(6*len(iters_to_plot),5))
    if len(iters_to_plot) == 1:
        axes = [axes]
    for ax, it in zip(axes, iters_to_plot):
        U = hist["U"][it]
        ax.scatter(U[0, :], U[1, :], alpha=0.5)
        xx, yy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
        zz = G_of_u(np.vstack([xx.ravel(), yy.ravel()]),t_grid).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[0], colors='red', linewidths=2)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"ALDI ensemble iter {it}")
        ax.set_xlabel("u1"); ax.set_ylabel("u2")
    plt.tight_layout()
    plt.show()


obs = np.array([0.01,0.1]) #, 0.1, 1
sizes = [500] #, 100, 1000


# ---------------------------
# Mains
# ---------------------------
if __name__ == "__main__":
  for ll in obs:
    for i in sizes:
      np.random.seed(42)
      evolution_time = 1
      t_grid = np.linspace(0, evolution_time, 10)  # Shape: (200,)
      # Ensemble size
      J = i
      # Number of ALDI iterations
      n_iter =50000 #+ 500

      # Observation and noise
      y = np.array([[0.0]])
      Gamma = np.array([[ll]])

      center = np.array([[-2], [0]])
      U0 = center + np.random.randn(2, J)*np.sqrt(0.5)
      # Add prior for stability
      #P0_inv = inv(np.cov(U0))#np.eye(2)  # Prior covariance inverse
      mu0 = np.zeros((2,1)) + center
      Sigma0 =np.eye(2)*0.5
      P0_inv = np.linalg.inv(Sigma0)
      # Initial ensemble (smaller spread for stability)



      # Run full gradient-free ALDI
      U_final, hist = run_aldi(U0, 0, t_grid, Gamma, P0_inv, mu0, n_iter=n_iter, dt=0.0001)

      x = np.linspace(-2,2,100)
      y = np.linspace(-2,2,100)
      xx,yy = np.meshgrid(x,y)
      U = np.vstack([xx.ravel(), yy.ravel()])
      zz = G_of_u(U, t_grid)
      zz = zz.reshape(xx.shape)


      # Plot ensemble evolution
      plt.figure(figsize=(10,10))
      plt.contour(xx,yy,zz, levels=[0], colors="r")
      plt.contourf(xx,yy,zz, cmap ="magma")
      plt.scatter(U_final[0,:] , U_final[1,:], s=10, alpha=0.5)


      string = "testFinal_HYP_" + "_itr_"+ str(n_iter) + "_size_" + str(J) + "_R_" + str(Gamma[0][0]) + "_Sigma_"+ str(0.1) +"G_free" + "_v_0p8_m2"
      np.save(string, U_final)
      plt.title(f"Ensemble size: {i} and R: {ll}")
      plt.legend()
      plt.savefig(string+ ".png")

      plt.clf()



