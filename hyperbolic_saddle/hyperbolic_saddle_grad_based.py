

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from tqdm import tqdm_notebook as tqdm

# """Writing plotting functions that will be used for rest of the script."""

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
def plot_ensemble_scatter(hist,T,  iters_to_plot=[0, -1], xlim=(-2,2), ylim=(-2,2), k=30.0):
    fig, axes = plt.subplots(1, len(iters_to_plot), figsize=(6*len(iters_to_plot),5))
    if len(iters_to_plot) == 1:
        axes = [axes]
    for ax, it in zip(axes, iters_to_plot):
        U = hist["U"][it]
        ax.scatter(U[0, :], U[1, :], alpha=0.5)
        xx, yy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
        zz = G_of_u(np.vstack([xx.ravel(), yy.ravel()]),T).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[0], colors='red', linewidths=4)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"ALDI ensemble iter {it}, Delta={k}")
        ax.set_xlabel("u1"); ax.set_ylabel("u2")
    plt.tight_layout()
    plt.show()

def plot_pf_trajectory(hist, k=30.0):
    plt.figure(figsize=(8,5))
    plt.plot(hist['pf_est'], marker='o', markersize=2, label="Estimated Pf (G_tilde)")
    #plt.plot(hist['frac_in_failure'], marker='x', markersize=2, label="Frac in failure domain")
    #plt.plot(hist['mean_G'], marker='^', markersize=2, label="Mean G_of_u")
    plt.xlabel("Iteration")
    plt.ylabel("Metrics")
    plt.title(f"Estimated Failure probability for Delta={k}")
    plt.grid(True)
    plt.legend()
    plt.show()

# """#Gradient based ALDI: Numerical implementation

# In this notebook the algorithm for Gradient-based algorithm is coded and explained block by block. The algorithm, is based on the following SDE,

# $$du^{(j)}_t = \biggl[- C(U_t) ∇_{ u^{(j)}} \Phi( u^{(j)} ) + \frac{d+1}{j} ( u^{(j)}_t) - m(U_t)\biggl] dt + \sqrt(2)C^{1/2}(U_t) dW^{(j)}(t)$$

# with,

# $$ \Phi = \frac{1}{2R}\tilde{G}(x)^2 - \ln\rho_0 (x)$$

# Here,
# * $U=(u^{1},..,u^{J})$
# * $m(U)= \frac{1}{J}\sum_{J=1}^{J}u^{(j)}$
# * $C(U)= \frac{1}{J}\sum_{J=1}^{J}\biggl(u^{(j)} - m(U)\biggl)\biggl(u^{(j)} - m(U)\biggl)^{T}$
# * $C^{1/2}= \frac{1}{\sqrt(J)}\biggl(U-m(U)1_J \biggl)$

# The above expressions are taken from the section 2.2 of the paper. $ \tilde{G}$ is the modified limit state function and  $\rho$ is the prior.
# """



# """#limit state function:
# The LSF defined below as G returns negative values, if the mean position (output from the dynamical system) after time T is within a radius "r" of the origin.

# $ G(U) = A(x_0,y_0) - r$

# where,
# $A(x_0,y_0)= \int^T_0 \frac{1}{T}(x(s)^2+y(s)^2)ds$

# $x(s)= x_0 \exp(-\lambda t)$
# $y(s)= y_0 \exp(\mu t)$
# """

# Limit state function

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

    # Clip to prevent overflow
    #r2 = np.clip(r2, 0, 1e2)

    # Mean r^2 over time for each particle
    mean_r2 = np.mean(r2, axis=0)  # Shape: (N,)

    # Return G = 0.5 - mean_r2
    return mean_r2-0.5  # Shape: (N,)




# """#Modified limit state function:
# For consistency w.r.t bayesian setting the limit state function is modified as,
# $\tilde{G(u)}:= max(0, G(u))$.
# """

# def G_tilde(u, T, k=1.0):
#   return np.maximum(0,  G_of_u(u, T))



# """In the section below we introduce a smooth version of the above Modified LSF function.

# $$ \tilde{G}_{\delta}(x) = \phi_{\delta}(G(x)) \cdot G(x) =
# \begin{cases}
# 0, & \text{if } x<0,\\
# \frac{G(x) \psi_\delta (G(x)) }{\psi_\delta (G(x)) + \psi_\delta (\delta - G(x))}, & \text{if } x \in [0, \delta],\\
# G(x) , & \text{if } x > \delta,
# \end{cases}
# $$
# """

def G_tilde(u,T, k=0.01):
    #return np.maximum(0,  G_of_u(u))
    x = G_of_u(u, T)
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

# """#Definition of potential:
# In Bayesian setting,
# $\rho_* = \frac{1}{Z} \biggl[\exp(-\frac{1}{2R}\tilde{G}(x)^2 )\biggl] \rho_0$

# In Aldi algorithm the probability density is expressed as,
# $\rho_* = Z^{-1} exp(-\phi(x))$

# where $\phi$ is defined as,
# $\phi(x)= \frac{1}{2R}\tilde{G}(x)^2 - ln \rho_0(x)$

# """


def rho_gen(x, mu, Sigma):
    d = len(mu)
    inv_Sigma = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_Sigma)
    diff = x - mu[:, None]
    exponent = -0.5 * np.einsum('ij,ik,jk->k', inv_Sigma, diff, diff)
    return norm_const * np.exp(exponent)

# --- Fixed Gaussian parameters ---


# def phi(U,R, k):
#   mu = np.array([0, 0])
#   Sigma = np.array([[0.2,0],
#                   [0, 0.2]])
#   Gt =  G_tilde(U, k)
#   rho = rho_gen(U, mu, Sigma) #(1/(2*np.pi*0.1))*np.exp((-(U[0]-3)**2 -(U[1]-3)**2 )/)
#   #rho = np.clip(rho,1e-6,1e+6)
#   return 1/(2*R) * Gt**2 - np.log(rho)




def phi(U,T,R, k=0.1):
  mu = np.array([-2.5, -2.5])
  Sigma = np.array([[0.8,0],
                  [0, 0.8]])

  Gt =  G_tilde(U,T, k)
  rho = rho_gen(U, mu, Sigma)
  #rho =(1/(2*np.pi))*np.exp((-U[0]**2 -U[1]**2 )/2)
  #rho = np.clip(rho,1e-6,1e+6)
  return 1/(2*R) * Gt**2 - np.log(rho)





# """#Gradient of the potential:

# The gradient is calculated using finite difference method with $\epsilon$ as spatial step size.
# """

def grad_PHI(U,T, R, k=0.1, eps=0.001):
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

        phi_plus = phi(U_plus, T, R, k)   # shape (J,)
        phi_minus = phi(U_minus, T, R, k) # shape (J,)

        grad[i,:] = (phi_plus - phi_minus) / (2*eps)

    return grad


# """#Aldi gradient step:

# In this step we solve the discretized version of the SDE introduced earlier in this notebook and in the script, in the aldi_gradient_step function.

# The failure calculator calculates the probability of failure per step by fitting a multimodal gaussian on the ensemble and then by doing important sampling from this new obtained distribution.

# At each step a multimodal gaussian distribution is fitted to the ensemble,
# referred as $P_{GMM}$. Then failure probability is calculated as following steps,


# 1.   Draw samples: $u_i \sim P_{GMM}$
# 2.   Calculate weight: $w_i = \frac{p(u_i) }{  P_{GMM(u_i)}}$, where $p(u_i)$ is the prior
# 3.   Probability: $P_f = \frac{ \sum 1_{G(u)\leq 0}(u_i)w_i }{\sum w_i}$




# The run_aldi_gradient function calls the above two functions and saves the important informations like ensemble at each step, failure probability at each step, number of particles in failure domain and mean-value of the LSF obtained from the ensemble at each step.


# """

# Gradient-based ALDI step
def aldi_gradient_step(U, T, y, Gamma, dt=0.001, k=1.0, alpha=1.0, eps=1e-6):
    D, N = U.shape
    m = np.mean(U, axis=1, keepdims=True)
    A = U - m

    # Gradient term
    grad_phi = grad_PHI(U,T, Gamma, k, eps)
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
def run_aldi_gradient(U0, T, y, Gamma, n_iter=9000, dt=0.01, k=30.0, alpha=10.0, grad_weight=1.0, data_weight=2.0):
    U = U0.copy()
    hist = { 'U': [], 'frac_in_failure': [], 'mean_G': []}
    for it in tqdm(range(n_iter), desc="ALDI iterations"):
        frac_in_failure = np.mean(G_of_u(U, T) <= 0)
        mean_G = np.mean(G_of_u(U, T))
        hist['U'].append(U.copy())
        hist['frac_in_failure'].append(frac_in_failure)
        hist['mean_G'].append(mean_G)
        if it % 100 == 0:
            print(f"Iter {it}: Frac in failure domain = {frac_in_failure:.4f}, Mean G = {mean_G:.4f}")
        U = aldi_gradient_step(U, T, y, Gamma, dt=dt, k=k, alpha=alpha)
    return U, hist

# Main
if __name__ == "__main__":
  for i in [ 100]: #, 1000, 10000
     for val in [0.001,0.01,0.1]:
      print(f"commencing for size {i} and {val}")
     #
      np.random.seed(42)
      evolution_time = 1
      t_grid = np.linspace(0, evolution_time, 10)  # Shape: (200,)
      J = i
      n_iter = 5000*64#50000
      # Initialize near failure boundary
      center = np.array([[-2.5], [-2.5]])
      U0 = center + np.random.randn(2, J)*np.sqrt(0.8)
      B = 0.001
      y = np.array([[0.0]])
      Gamma = np.array([[val]])  # Larger Gamma for stability
      DT = 0.001/32#0.00001
      U_final, hist = run_aldi_gradient(U0, t_grid, y, Gamma, n_iter=n_iter, dt=DT, k=B)

      #plot_ensemble_scatter(hist,t_grid ,iters_to_plot=[0, n_iter//4, n_iter//2, n_iter-1], k=B)

      #string = f"Delta_varuation_HYP_itr_{n_iter}_size_{J}_R_{Gamma[0][0]}_Delta_{0.001}_evt_{evolution_time}" 
      string = f"Final_Hyp_" + "_itr_"+ str(n_iter) + f"size_{i}_R_{val}_Delta_{B}_evt_{evolution_time}_v_0p8_m2" 
      Uc = np.save(string, hist["U"])

       
      x = np.linspace(-2,2,100)
      y = np.linspace(-2,2,100)

      xx, yy = np.meshgrid(x,y)  
      U = np.vstack([xx.ravel(), yy.ravel()])
      zz = G_of_u(U, t_grid)
      zz = zz.reshape(xx.shape)
      
      plt.figure(figsize=(10,10))
      #plt.contourf(xx,yy,zz)
      plt.contour(xx,yy,zz, levels=[0], colors='red', linewidths=4 )
      plt.scatter(U_final[0], U_final[1])
      plt.title(f"Ensemble size: {i} and R: {val}")
      plt.legend()
      plt.savefig(string+ ".png")

      plt.show()