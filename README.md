# Gradient-Based and Gradient-Free ALDI for Failure Probability Estimation

This repository implements **gradient-based** and **gradient-free** variants of an ALDI (Affine-Invariant Langevin Dynamics Inference) algorithm to estimate failure probabilities in a dynamical system using ensembles of particles evolved by stochastic dynamics. The algorithms are experimented on three specific problems:
  - Algebraic Convex problem
  - Hyperbolic Saddle problem
  - 6 dimensional Point vortex interaction problem

In all three examples the ALDI distribution evolves in the phase-space to capture the rare-event set. The rare-events are defined as the zero-level set of the limit-state-function, in a bayesian-inverse-problem context. A smooth and non-smooth version of this LSF is presented to overcome differentiability contrainst. The repository builds on the paper Schillings et al 2025 (https://arxiv.org/abs/2601.00107).  

---

## Gradient-Based ALDI

ALDI is formulated as an interacting particle system, which in presence of a smooth potential function can be expressed as a system of Ito-type stochastic differential equations. The gradient-based implementation evolves an ensemble according to a discretized SDE with drift and diffusion terms constructed from a potential function under the influence of standard brownian noise, given as,

$dx_t^{(j)} =\left[ - C(X_t) \nabla_{x^{(j)}} \Phi \left(x_t^{(j)}\right) + \frac{d+1}{J}\left(x_t^{(j)} - m(X_t)\right) \right] dt + \sqrt{2}\, C^{1/2}(X_t)\, dW_t^{(j)}$

where,

$m(X) := \frac{1}{J} \sum_{j=1}^{J} x^{(j)}$

$C(X) := \frac{1}{J} \sum_{j=1}^{J} \left(x^{(j)} - m(X)\right)\left(x^{(j)} - m(X)\right)^{\top}$

$C^{1/2}(X) := \frac{1}{\sqrt{J}} \left( X - m(X)\mathbf{1}_J \right)$

Below different functions of the code is explained as building blocks. For more details, look into section 2.2 of Schillings et al, 2025.
### Main Components

- `G_of_u(u, t_grid)`  
  The limit-state function depicted as $G(\cdot)$, is a map from $G:\mathbb{R}^D \rightarrow \mathbb{R}$. The outcome of which defines a rare-event in bayesian sense. Definition of rare-event is given as, $G(\cdot) \leq 0$.

- `G_tilde(u, T, k)`  
  To deal with the non-smooth potential, a piecewise approximation of the original limit state function is made in an interval $[0,\delta]$.
  
$G_\delta(x) = 0, \quad \text{if } G(x) < 0$

$G_\delta(x) = \dfrac{G(x)\,\psi_\delta(G(x))}{\psi_\delta(G(x)) + \psi_\delta(\delta - G(x))}, \quad \text{if } G(x) \in [0, \delta]$

$G_\delta(x) = G(x), \quad \text{if } G(x) > \delta$

Where, $\psi_\delta$ can be thought of as a mollifier.The important properties regarding the above approximation is that it,  preserves the zero-level-sets for all $\delta \geq 0$; and as $\tilde{G}_\delta \rightarrow G$ as $\delta \rightarrow 0$.


- `rho_gen(x, mu, Sigma)`  
  Evaluates a **multivariate Gaussian prior density**:
    $$\rho_0(x) = \mathcal{N}(x;\mu,\Sigma)$$
  used to regularize the ensemble and encode prior information. It is assumed that $\rho_0$ is positive and conitnuous in the state-space.

- `phi(U, T, R, k)`  
  Defines the **potential function**:
    $$\Phi(x) = \frac{1}{2R}\tilde{G}(x)^2 - \ln \rho_0(x)
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
    - Forward model values `P = is_failure(U, t_grid)`.
    - Output mean `p_mean` and anomalies `B = P - p_mean`.
    - Empirical covariance $C$.
    - Covariance-based square-root term $C_\text{half}$ (using anomalies).
  - Evaluates `grad_PHI` and constructs the **drift**:
    $-C \nabla \Phi(U) + \frac{D+1}{N}(U - m)$$
  - Adds noise:
    $$\text{noise} \propto C^{1/2} \xi$$
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

In addition to the gradient-based implementation, the repository includes a **gradient-free ALDI** variant that does not require explicit gradients of $\Phi$ or the LSF. The approximated version is given as,
$dx_t^{(j)} = \Big[- \frac{1}{R} D(X_t) \,\tilde{G}\big(x_t^{(j)}\big)+ \frac{1}{P_0} C(X_t) \big(x_t^{(j)} - m_0\big)+ \frac{d+1}{J} \big(x_t^{(j)} - m(X_t)\big)
\Big] dt + \sqrt{2}\, C^{1/2}(X_t) \, dW_t^{(j)}$

$D(X) := \frac{1}{J} \sum_{j=1}^{J} \left( x^{(j)} - m(X) \right) \left( G(x^{(j)}) - m(G(X)) \right)^{\top}$

$m(G(X)) := \frac{1}{J} \sum_{j=1}^{J} G(x^{(j)})$

where $D$ is the cross-correlation matrix.  For more details regarding formulation and use of gradient free version, look into section 2.4 of the Schillings et al.2025. For the theoretical origin and  analysis of the gradient-free version refer to Reich et al 2020.


### Core Idea

- The gradient-free method:
  - Capable using the original $G_tilde$ with the max function instead of the modified function that is piecewise-smooth.
  - Uses the **empirical cross-covariance** between particle states and their outputs to build a drift toward observations.
  - Incorporates a **Gaussian prior** to stabilize dynamics and control spread.
- This is particularly useful when:
  - The LSF or model is a **black box**.
  - Analytical or numerical gradients are expensive or unavailable.

### Main Components

- `G_of_u(u, t_grid)`  (same as before)

- `is_failure(u, t_grid)`  
  - Returns $\max(0, G(u))$, a non-negative failure response used as the observation/forward model in the gradient-free update.

- `aldi_gradient_free_step(U, y, t_grid, Gamma, P0_inv, mu0, dt=0.1, eps=1e-12)`  
  Performs **one gradient-free ALDI step**:
  - Computes:
    - Ensemble mean `m` and anomalies $A = U - m 1_N$.
    - Forward model values `P = is_failure(U, t_grid)`.
    - Output mean `p_mean` and anomalies `B = P - p_mean`.
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

## Limit State Functions in the Three Examples

This section summarizes how the limit state functions $$G$$ are defined in each of the three test problems. The failure set is always characterized by $G(u) \le 0$, and the ALDI algorithms are used to drive ensembles toward these rare-event regions.

### Convex Algebraic Problem

In the convex problem, $u = (u_1, u_2)$ is a 2D parameter and $G$ is a quadratic-plus-linear function:
$G(u) = 0.1(u_1 - u_2)^2 - \frac{1}{\sqrt{2}}(u_1 + u_2) + 2.5.$

Failure is defined by $G(u) \le 0$, and the corresponding clipped “failure response” used in the code is
`is_failure(u) = max(0, G(u))`


This creates a convex rare-event region in the $(u_1,u_2)$ plane.

### Hyperbolic Saddle Problem

In the hyperbolic saddle problem, $u = (u_1, u_2)$ represents initial conditions for a 2D dynamical system with one stable and one unstable direction, integrated over a time grid `t_grid`.

The trajectories are $\(x(t) = u_1 e^{-\alpha t}\)$ and $\(y(t) = u_2 e^{\beta t}\)$ with $\alpha = 1$ and $\beta = 1$ in the implementation. For each particle,
$r(t)^2 = x(t)^2 + y(t)^2.$

The limit state function is defined as the time-averaged squared radius minus a threshold:
$G(u) = \frac{1}{|t_{\text{grid}}|} \sum_{t \in t_{\text{grid}}} r(t)^2 - 0.5.$

Failure corresponds to $G(u) \le 0$, i.e., trajectories that on average remain within a radius satisfying roughly $r^2 \le 0.5$. The associated clipped response is
`is_failure(u, t_grid) = (0, G(u))`.

This produces a nontrivial rare-event region shaped by the saddle dynamics in phase space.

### 6D Point Vortex Interaction Problem

In the vortex interaction problem, $u \in \mathbb{R}^6$ is a 6D control/noise vector that perturbs the motion of three interacting point vortices in 2D. The limit state function is built from a geometric constraint on the vortex configuration.

#### Underlying dynamics

Three vortices with circulations $\Gamma_1 = 1$, $\Gamma_2 = 1$, $\Gamma_3 = -2$ interact via a Biot–Savart–type vector field (implemented by the functions `Fx1`, `Fx2`, `Fx3`, `Fy1`, `Fy2`, `Fy3`). The routine `Euler_alt` integrates the vortex trajectories forward in time using an Euler–Maruyama scheme, with additive noise driven by $u$.

#### Initialization near a special configuration

The function `yield_X2_X3` constructs initial positions for vortices 2 and 3 from vortex 1 such that the three vortices start near a special geometric configuration (approximately equilateral) with side length $L$ determined by the circulations and Hamiltonian parameters.

#### Geometric functional $A$

At any time, for positions $(x_1,x_2,x_3,y_1,y_2,y_3)$, the functional $A$ measures how far the configuration is from this target shape.

Define the edge vectors: $v_{21} = (x_2,y_2) - (x_1,y_1)$, $v_{31} = (x_3,y_3) - (x_1,y_1)$, $v_{32} = (x_3,y_3) - (x_2,y_2)$
and their norms $\lVert v_{21} \rVert$, $\lVert v_{31} \rVert$, $\lVert v_{32} \rVert$. From these, one computes cosines of angles (e.g. $\cos A$, $\cos B$) and compares them to $0.5$, the value corresponding to a $60^\circ$ angle.

The functional is then
$A = \lvert \cos A - 0.5 \rvert + \lvert \cos B - 0.5 \rvert + \left\lvert \frac{1}{3} \bigl( \lVert v_{21} \rVert + \lVert v_{31} \rVert + \lVert v_{32} \rVert \bigr) - L \right\rvert.$

This penalizes deviations from an approximately equilateral configuration with perimeter-controlled side length $L$.

#### Time-averaged geometric deviation

The function `A_mean` evaluates $A$ along each trajectory and averages over time:
$\bar{A}(u) = \text{mean over time of } A(\text{vortex positions}(t; u)).$

#### Limit state function for vortices

The atmospheric/vortex limit state function is defined as
$G(u) = \bar{A}(u) - \text{threshold},$
with a chosen threshold (e.g. $0.25$) and specific time-step $\tau$ and final time $T_f$ inside `G_atm_alt` / `G_of_u`.

Failure corresponds to $G(u) \le 0$, i.e., those noise/control realizations $u$ that keep the three-vortex system close (on average over time) to the target configuration.

---

## Notes

- Analytical saddle-point or other closed-form derivations are **intentionally omitted**; the focus is on numerical ALDI implementations.
- Both variants rely on **ensemble statistics** and are suitable for reliability analysis and rare-event estimation in systems defined via black-box forward models.
- Random seeds are set (`np.random.seed(42)`) for reproducible experiments.
- You can extend the plotting utilities or post-process the saved `.npy` files in separate notebooks for additional diagnostics (e.g., convergence of estimated failure probabilities or comparison between gradient-based and gradient-free behavior).


## References

1.  Chakraborty, D., Harris, R., Klein, R., Olicón-Méndez, G., Reich, S., & Schillings, C. (2025). *Affine Invariant Langevin Dynamics for rare-event sampling*. arXiv:2601.00107. [https://arxiv.org/abs/2601.00107]
2.  Garbuno-Inigo, A., Nüsken, N., & Reich, S. (2020). *Affine Invariant Interacting Langevin Dynamics for Bayesian Inference*. SIAM Journal on Applied Dynamical Systems, 19(3), 1633–1658. https://doi.org/10.1137/19M1304891


