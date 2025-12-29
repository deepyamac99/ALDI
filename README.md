# Gradient-Based and Gradient-Free ALDI for Failure Probability Estimation

This repository implements **gradient-based** and **gradient-free** variants of an ALDI (Affine-Invariant Langevin Dynamics Inference–type) algorithm to estimate failure probabilities in a dynamical system using ensembles of particles evolved by stochastic dynamics. The algorithms are experimented on three specific problems:
  - Algebraic Convex problem
  - Hyperbolic Saddle problem
  - 6 dimensional Point vortex interaction problem

In all three examples the ALDI distribution evolves in the phase-space to capture the rare-event set.

---

## Gradient-Based ALDI

The gradient-based implementation evolves an ensemble according to a discretized SDE with drift and diffusion terms constructed from a potential function.

### Core Idea

- The SDE uses:
  - **Empirical mean** $m(U)$ and **covariance** $C(U)$ of the ensemble.
  - A **potential** $\Phi(u)$ combining a piecewise smoothed limit state function $\tilde{G}$ and a Gaussian prior.
  - A **gradient term** $-C(U)\nabla \Phi(u^{(j)})$ driving the ensemble toward regions of interest (e.g., failure domain).
- Noise is added using a covariance-based construction, ensuring exploration and approximate affine invariance.

### Main Components

- `G_of_u(u, t_grid)`  
  Defines the **limit state function (LSF)** based on a 2D dynamical system:
  - The state $(x(t), y(t))$ is obtained by exponential decay/growth from the initial condition $u = (u_1, u_2)$ over a time grid `t_grid`.
  - The function computes the time-averaged squared radius $x(t)^2 + y(t)^2$ and returns:
    $$
    G(u) = \text{mean}_t(x(t)^2 + y(t)^2) - 0.5
    $$
  - Failure corresponds to $G(u) \le 0$.

- `G_tilde(u, T, k)`  
  Implements a **smoothed version** of the modified LSF:
  - Uses a smooth transition function $\phi_\delta$ so that:
    - $\tilde{G}_\delta(u) \approx 0$ for negative $G(u)$.
    - $\tilde{G}_\delta(u) \approx G(u)$ away from the transition region.
  - This improves differentiability and numerical stability of the potential.

- `rho_gen(x, mu, Sigma)`  
  Evaluates a **multivariate Gaussian prior density**:
    $$
    \rho_0(x) = \mathcal{N}(x;\mu,\Sigma)
    $$
  used to regularize the ensemble and encode prior information.

- `phi(U, T, R, k)`  
  Defines the **potential function**:
    $$
    \Phi(x) = \frac{1}{2R}\tilde{G}(x)^2 - \ln \rho_0(x)
    $$
  where:
  - $R$ is a scaling parameter.
  - $\tilde{G}(x)$ is the smoothed LSF.
  - $\rho_0(x)$ is the Gaussian prior density.

- `grad_PHI(U, T, R, k=0.1, eps=0.001)`  
  Computes the **gradient of the potential** using finite differences:
  - For each component of $u$, evaluates $\Phi(u \pm \varepsilon e_i)$.
  - Approximates $\partial \Phi / \partial u_i$ via central differences.
  - Returns a gradient matrix with shape `(2, J)` for all particles.

- `aldi_gradient_step(U, T, y, Gamma, dt, k, alpha, eps)`  
  Performs **one gradient-based ALDI step**:
  - Computes:
    - Ensemble mean `m` and anomalies $A = U - m 1_N$.
    - Forward model values $P = \text{is\_failure}(U, t\_grid)$.
    - Output mean `p_mean` and anomalies $B = P - p_\text{mean}$.
    - Empirical covariance $C$.
    - Covariance-based square-root term $C_\text{half}$ (using anomalies).
  - Evaluates `grad_PHI` and constructs the **drift**:
    $$
    \text{drift} = -C \nabla \Phi(U) + \frac{D+1}{N}(U - m)
    $$
  - Adds noise:
    $$
    \text{noise} \propto C^{1/2} \xi
    $$
    where $\xi$ is standard Gaussian.
  - Uses an Euler–Maruyama update for each particle.

- `run_aldi_gradient(U0, T, y, Gamma, n_iter, dt, k, alpha, grad_weight, data_weight)`  
  Runs the **full gradient-based ALDI algorithm**:
  - Iteratively calls `aldi_gradient_step`.
  - Tracks:
    - `hist['U']`: ensemble snapshots.
    - `hist['frac_in_failure']`: fraction of particles with $G(u) \le 0$.
    - `hist['mean_G']`: average LSF value.
  - Prints diagnostics every 100 iterations.

### Plotting Utilities (Gradient-Based)

- `plotter(xx, yy, zz)`  
  - 2D contour plot of the LSF and 3D surface view on a grid.
- `plot_ensemble_scatter(hist, T, iters_to_plot, xlim, ylim, k)`  
  - Shows ensemble positions at selected iterations overlaid with the $G(u)=0$ contour.
- `plot_pf_trajectory(hist, k)`  
  - Plots the evolution of estimated failure-related metrics per iteration.

### Running the Gradient-Based Version

In the `__main__` block:

- Typical configuration:
  - `evolution_time = 1`
  - `t_grid = np.linspace(0, evolution_time, 10)`
  - `J`: ensemble size, e.g. `100`
  - `n_iter`: total iterations, e.g. `5000*64`
  - `center = np.array([[-2.5], [-2.5]])`
  - `U0 = center + np.random.randn(2, J)*np.sqrt(0.8)`
  - `B`, `Gamma`, `DT`: hyperparameters for smoothing, observation noise scale, and time step.
- After calling `run_aldi_gradient`, the code:
  - Saves the ensemble history as a `.npy` file.
  - Plots the failure boundary and final ensemble.
  - Stores the figure as `.png` with a descriptive filename.

---

## Gradient-Free ALDI Variant

In addition to the gradient-based implementation, the repository includes a **gradient-free ALDI** variant that does not require explicit gradients of $\Phi$ or the LSF. Instead, it uses ensemble covariances between parameters and forward model outputs to construct data-driven drifts, similar in spirit to ensemble Kalman methods.

### Core Idea

- The gradient-free method:
  - Treats the **forward model** as the smoothed failure response `is_failure(u, t_grid)`.
  - Uses the **empirical cross-covariance** between particle states and their outputs to build a drift toward observations.
  - Incorporates a **Gaussian prior** to stabilize dynamics and control spread.
- This is particularly useful when:
  - The LSF or model is a **black box**.
  - Analytical or numerical gradients are expensive or unavailable.

### Main Components

- `G_of_u(u, t_grid)`  
  Same LSF definition as in the gradient-based implementation:
  - Computes trajectories over `t_grid`, evaluates $x(t)^2 + y(t)^2$, averages over time, and subtracts `0.5`.
  - Failure: $G(u) \le 0$.

- `is_failure(u, t_grid)`  
  - Returns $\max(0, G(u))$, a non-negative failure response used as the observation/forward model in the gradient-free update.

- `aldi_gradient_free_step(U, y, t_grid, Gamma, P0_inv, mu0, dt=0.1, eps=1e-12)`  
  Performs **one gradient-free ALDI step**:
  - Computes:
    - Ensemble mean `m` and anomalies $A = U - m 1_N$.
    - Forward model values $P = \text{is\_failure}(U, t\_grid)$.
    - Output mean `p_mean` and anomalies $B = P - p_\text{mean}$.
  - Builds empirical covariances:
    - State covariance $C = \frac{1}{N} A A^\top + \varepsilon I$.
    - Cross-covariance $C_{uG} = \frac{1}{N} A B^\top$.
  - Uses:
    - Observation noise inverse $R_\text{inv} = \Gamma^{-1}$.
    - Prior inverse covariance `P0_inv` and prior mean `mu0`.
  - For each particle:
    - **Data drift**: $- C_{uG} R^{-1} (G_i - y)$.
    - **Prior drift**: $- C P_0^{-1} (u_i - \mu_0)$.
    - **Finite-$N$ correction**: $\frac{D+1}{N}(u_i - m)$.
    - Noise via anomalies: $\sqrt{2dt}\, C^{1/2} \xi$ using `C_half = 1/√(N-1) * A`.
  - Updates particles using Euler–Maruyama.

- `run_aldi(U0, y, t_grid, Gamma, P0_inv, mu0, n_iter, dt)`  
  Runs the **full gradient-free ALDI**:
  - Iteratively calls `aldi_gradient_free_step`.
  - Stores:
    - `hist['U']`: ensemble snapshots.
    - `hist['frac_in_failure']`: fraction with $G(u) \le 0$.
    - `hist['mean_G']`: mean LSF.
  - Prints diagnostics every 100 iterations.

- `plot_ensemble_scatter(hist, t_grid, iters_to_plot, xlim, ylim)`  
  - Visualizes the ensemble at chosen iterations over the failure contour $G(u)=0$.

### Running the Gradient-Free Version

In its `__main__` block:

- Example configuration:
  - `obs = np.array([0.01, 0.1])` for different noise levels.
  - `sizes = [500]` for ensemble sizes.
  - `evolution_time = 1`, `t_grid = np.linspace(0, evolution_time, 10)`.
  - `J = i` (ensemble size), `n_iter = 50000`.
  - Observation setup:
    - `y = np.array([[0.0]])` (target failure response).
    - `Gamma = np.array([[ll]])` (scalar observation covariance).
  - Prior:
    - `center = np.array([[-2], [0]])`.
    - `U0 = center + np.random.randn(2, J)*np.sqrt(0.5)`.
    - `mu0 = center` and `Sigma0 = 0.5 * I`.
    - `P0_inv = inv(Sigma0)`.
  - Time step: `dt = 0.0001`.

- After running `run_aldi`:
  - The script generates:
    - A contour plot of `G(u)` with $G(u)=0$ in red.
    - A filled contour background and final ensemble scatter.
  - Saves:
    - Final ensemble: `testFinal_HYP__itr_<n_iter>_size_<J>_R_<Gamma>_Sigma_0.1G_free_v_0p8_m2.npy`.
    - Plot: same prefix with `.png`.

---

## Installation

Clone the repository and install dependencies using a Python environment (e.g., `venv` or Conda):


---

## Usage Summary

- **Gradient-based ALDI**:
  - Script: your gradient-based file (e.g., `gradient_based_aldi.py`).
  - Adjust parameters in the `__main__` section (ensemble size, time step, number of iterations, prior parameters).
  - Run:
    ```
    python gradient_based_aldi.py
    ```
  - Inspect saved `.npy` and `.png` outputs.

- **Gradient-free ALDI**:
  - Script: your gradient-free file (e.g., `gradient_free_aldi.py`).
  - Configure `obs`, `sizes`, prior, and `n_iter` in `__main__`.
  - Run:
    ```
    python gradient_free_aldi.py
    ```
  - Analyze final ensembles and plots for different noise levels and ensemble sizes.

---

## Notes

- Analytical saddle-point or other closed-form derivations are **intentionally omitted**; the focus is on numerical ALDI implementations.
- Both variants rely on **ensemble statistics** and are suitable for reliability analysis and rare-event estimation in systems defined via black-box forward models.
- Random seeds are set (`np.random.seed(42)`) for reproducible experiments.
- You can extend the plotting utilities or post-process the saved `.npy` files in separate notebooks for additional diagnostics (e.g., convergence of estimated failure probabilities or comparison between gradient-based and gradient-free behavior).
