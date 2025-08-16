# MIT License

# Copyright (c) 2025 Dmitri Ivanov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This script simulates k_E3 catalytic rates for heterobifunctional degraders (PROTACs) 
by numerically solving a system of three mass-balance equations describing binding equilibria.

Features:
- Simulates k_E3 rates both with and without added experimental noise.
- Saves results as two separate .mat files for use by `protac_fit_sim_rates.py`.
- Generates theoretical plots of k_EDS vs D_t and k_EDS vs S_t.

Convergence Monitoring:
- Tracks potential convergence issues in `fsolve()` and prints warning messages.
- Warnings can generally be ignored unless discontinuities appear in the plots.
- If discontinuities occur, the reported (D_t, S_t) indices in the warnings can help 
  locate problematic parameter combinations.
- Such issues may be resolved by adjusting parameter values or the error tolerance (`tol`).
"""
# ### =============================================================================
# ### SECTION 1: Initialization
# ### =============================================================================

# ### --- Import libraries and commands ---

import numpy as np
from scipy.optimize import least_squares, fsolve
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import savemat


### --- Set parameters ---

# output file name
mat_file_name_1="file_name_protac_simulated_rates_no_noise.mat"
mat_file_name_2="file_name_protac_simulated_rates_with_noise.mat"

# Parameter values used for calculating simulated rates.
# Use the following units:
#   K_ED, K_SD, K_EDS in nM
#   k_cat_over_K_m in (1 / (mM·s)); for dBET1, BRD4 and cereblon system k_cat_over_K_m ~ 30 (1 / (mM·s))
K_ED = 100
K_SD = 100
K_EDS = 100
k_cat_over_K_m = 10

#fractional level for simulated experimental noise
noise=0.03

#background rate (degrader- and substrate-indepenent consumption of the E2-Ub substrate by the E3 ligase)
background_rate=0.5

# D_t concentrations (nM) for which k_E3 rates are simulated (use the same array as in the ..._glue_fit_rates.py script that is being tested)
D_t = np.array([0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

# E_t and S_t concentrations (nM) for which k_E3 rates are simulated (use the same arrays as in the ..._glue_fit_rates.py script that is being tested)
E_t = np.array([ 40, 40, 40, 40, 40, 40, 40])
S_t = np.array([ 0, 50, 100, 200, 400, 800, 1600])

# If there is a systematic bias in nominal total concentrations (E_t, S_t, D_t),
# apply a correction coefficient here (1.0 = no correction).
coef_1=1

# output fig1 name
fig1_name="file_name_protac_simulated_rates_vs_D_t.svg"

# output fig2 name
fig2_name="file_name_protac_simulated_rates_vs_S_t.svg"

# ### =============================================================================
# ### SECTION 2: Function Definitions
# ### =============================================================================

# ----- Compute [EDS] and calculate the catalytic rate k_EDS = k_cat_over_K_m * [EDS}

# This is the main function of the script.
# It numerically solves for the concentrations of three unknowns—[EDS], free [D] and free [S]—
# given total concentrations E_t, S_t, and D_t, and parameter values K_ED, K_SD, K_EDS, and k_cat_over_K_m.
# The solution is obtained by solving the system of mass conservation equations
# that describe binding equilibria of heterobifunctional degraders (protacs) (see TPD_protac_system).
# The vector of unknowns is represented by array x: x[0] = [EDS], x[1] = [D], x[2] = [S].

def rate_vs_D_t_fsolve_protac(K_ED, K_SD, K_EDS, E_t, S_t, D_t, k_cat_over_K_m, coef_1):
    kcat_Km_per_nM_per_h=k_cat_over_K_m*3600/1000000  # Convert rate constant to per nM per hour
    rate = np.zeros((D_t.size,S_t.size)) # Initialize array for calculated rates with the same shape as k_E3 (experimental rates)
    tol = 0.01 # Tolerance for numerical error to allow small boundary violations

    # Generate initial guess x0 with random values within valid ranges
    x0=initial_guess(E_t[0],S_t[0],D_t[0])

    for j in range(S_t.size):
        for i in range(D_t.size):
            if not ((x0[0] <= min(E_t[j],S_t[j]* coef_1,D_t[i])) and (x0[0] >= 0) and
                    (x0[1] <= min(E_t[j],S_t[j]* coef_1)) and (x0[1] >= 0) and
                    (x0[2] <= S_t[j] * coef_1) and (x0[2] >= np.maximum(S_t[j] - E_t[j], 0))):
                x0=initial_guess(E_t[j],S_t[j],D_t[i])
            while True:
                # Define the system of equations to be solved
                fun = lambda x: TPD_protac_system(x, K_ED, K_SD, K_EDS, E_t[j], S_t[j], D_t[i], coef_1)
                # Solve for the root using fsolve()
                x, _, ier, _ = fsolve(fun, x0, full_output=True)
                # Define error tolerance for root validation
                error = tol * min([l for l in [E_t[j],S_t[j]* coef_1,D_t[i]] if l != 0])
                
                # Check whether the root is within acceptable bounds
                if not ((x[0] <= min(E_t[j],S_t[j]* coef_1,D_t[i]) + error) and (x[0] >= -error) and
                       (x[1] <= D_t[i] + error) and (x[1] >= (np.maximum(D_t[i] - (S_t[j] + E_t[j]),0)-error)) and
                       (x[2] <= S_t[j] * coef_1 + error) and (x[2] >= np.maximum(S_t[j] - D_t[i],0)-error)):
                    # comment out to silence the warning
                    print(f'[{i},{j}]: invalid root, retrying with new initial guess')
                    x0=initial_guess(E_t[j],S_t[j],D_t[i])
                    continue

                # Check whether fsolve() has converged normally
                if ier != 1:
                    # comment out to silence the warning
                    print(f'[{i},{j}]: fsolve() did not converge, retrying with new initial guess')
                    x0=initial_guess(E_t[j],S_t[j],D_t[i])
                    continue
                
                # Calculate the catalytic rate k_EDS
                rate[i, j] = kcat_Km_per_nM_per_h * x[0]
                x0=x
                break
    return rate

def TPD_protac_system(x, K_ED, K_SD, K_EDS, E_t, S_t, D_t, coef_1):
    # Initialize the output array F
    F = np.zeros(3)
    
    # The vector of unknowns is x: x[0] = [EDS], x[1] = free [D], x[2] = free [S]

    # Define the system of equations based on mass conservation and equilibrium binding
    F[0] = x[0] * K_ED * K_EDS + x[0] * x[1] * K_EDS + x[0] * x[1] * x[2] - E_t * x[1] * x[2]
    F[1] = x[2] + (x[1] * x[2]) / K_SD + x[0] - S_t * coef_1
    F[2] = x[1] * x[2] + x[0] * K_EDS + (x[1] * x[2]**2) / K_SD + x[0] * x[2] - D_t * x[2]
    
    return F

def initial_guess(E_t, S_t, D_t):
    # x0[0]: random value between 0 and min(E_t, S_t, D_t)
    x0_0 = np.random.rand() * min(E_t, S_t, D_t)
    
    # x0[1]: random value between max(D_t - (S_t + E_t), 0) and D_t
    lower_bound_1 = np.maximum(D_t - (S_t + E_t), 0)
    upper_bound_1 = D_t
    x0_1 = np.random.rand() * (upper_bound_1 - lower_bound_1) + lower_bound_1

    # x0[2]: random value between max(S_t - D_t, 0) and S_t
    lower_bound_2 = np.maximum(S_t - D_t, 0)
    upper_bound_2 = S_t
    x0_2 = np.random.rand() * (upper_bound_2 - lower_bound_2) + lower_bound_2

    return np.array([x0_0, x0_1, x0_2])


colors_rgb = [
    # (0, 0, 0),    
    (0.96, 0.043, 0.043),
    # (0.98, 0.62, 0.055),
    (0.98039216, 0.88235294, 0.05490196),
    # (0.77254902, 0.69803922, 0.07058824),
    # (0.54901961, 0.83921569, 0.09411765),
    (0.11372549, 1.        , 0.55686275),
    (0.0627451 , 0.62352941, 0.34509804),
    # (0.04705882, 0.6       , 0.69803922),
    (0.0745098 , 0.85882353, 1.        ),
    (0.16078431, 0.10588235, 1.        ),
    (0.40784314, 0.08627451, 0.61960784)
]



# ### =============================================================================
# ### SECTION 3: Calculate theoretical rates and output into two .mat files with and without noise added
# ### =============================================================================

# Initialize data storage
sim_rates_1 = np.zeros((D_t.size, S_t.size))
sim_rates_2 = np.zeros((D_t.size, S_t.size))
sim_rates_plus_noise = np.zeros((D_t.size, S_t.size))

# Call the external function rate_vs_D_t_fsolve to generate simulated k_EDS rates for E_t, S_t and D_t array values using molecular glue equations.
sim_rates_1 = rate_vs_D_t_fsolve_protac(K_ED, K_SD, K_EDS, E_t, S_t, D_t, k_cat_over_K_m, coef_1)

# generate simulated k_E3 values by adding background_rate to k_EDS values
sim_rates_2 = sim_rates_1 + background_rate

# add random noise to k_E3 values at the level specified by noise
sim_rates_plus_noise = np.random.normal(loc=sim_rates_2, scale=np.abs(sim_rates_2) * noise)

# Save data
savemat(mat_file_name_1, {'k_E3': sim_rates_2})
savemat(mat_file_name_2, {'k_E3': sim_rates_plus_noise})

# ------- Plot experimental and fitted k_EDS vs D_t for different S_t concentrations

plt.figure(1, figsize=(6, 6))

# Define D_t range for plotting the fitted curves
D_t_range = np.logspace(np.log10(np.min(D_t[D_t != 0])) - 0.2, np.log10(np.max(D_t[D_t != 0])) + 0.2, 100)
# Compute theoretical curves for all S_t values
fitted_curves = rate_vs_D_t_fsolve_protac(K_ED, K_SD, K_EDS, E_t, S_t, D_t_range, k_cat_over_K_m, coef_1)
rate_data_values = rate_vs_D_t_fsolve_protac(K_ED, K_SD, K_EDS, E_t, S_t, D_t, k_cat_over_K_m, coef_1)

# Plot experimental points and corresponding fitted curves
for i in range(S_t.size):
    plt.scatter(D_t[:], rate_data_values[:, i], color=colors_rgb[i], facecolors=colors_rgb[i], label=f'S_t = {S_t[i]/1000} uM', clip_on=False)
    plt.plot(D_t_range, fitted_curves[:, i], '-', color=colors_rgb[i], clip_on=False)

plt.xlabel('D_t (nM)')
plt.ylabel('k_EDS (1/hour)')
plt.legend()
plt.gca().set_yscale('linear')
plt.gca().set_xscale('log')
# plt.gca().set_xlim([1, 10000])
# plt.gca().set_ylim([-0.1, 4])
plt.savefig(fig1_name, format="svg")
# plt.show()

# ------- Plot k_EDS vs S_t for minimal D_t and for D_t at which rate reaches the maximum value
# ------- This helps visualize how much the molecular glue enhances E3–substrate affinity

plt.figure(2, figsize=(6, 6))

# Define S_t range for plotting
S_t_range = np.linspace(0, S_t.max() * 1.1, 100)

# Generate E_t array of same size as S_t_range (assumes constant E_t across all S_t)
E_t_range = np.linspace(E_t[0], E_t[0], 100)


# Compute fitted curves for minimal D_t and for D_t at which rate reaches the maximum value
min_max_D_t_rate_fits = rate_vs_D_t_fsolve_protac(K_ED, K_SD, K_EDS, E_t_range, S_t_range, np.array([D_t.min(), D_t[np.unravel_index(np.argmax(rate_data_values), rate_data_values.shape)[0]]]), k_cat_over_K_m, coef_1)

# Plot experimental data and fitted curves for min and max D_t
plt.scatter(S_t[:]/1000, rate_data_values[D_t.min(),:], color=(0,0,0), facecolors=(0,0,0), label=f'D_t = {np.min(D_t)} nM', clip_on=False)
plt.plot(S_t_range/1000, min_max_D_t_rate_fits[0,:], '-', color=(0,0,0), clip_on=False)
plt.scatter(S_t[:]/1000, rate_data_values[np.unravel_index(np.argmax(rate_data_values), rate_data_values.shape)[0],:], color=(1,0,0), facecolors=(1,0,0), label=f'D_t = {D_t[np.unravel_index(np.argmax(rate_data_values), rate_data_values.shape)[0]]} nM', clip_on=False)
plt.plot(S_t_range/1000, min_max_D_t_rate_fits[1,:], '-', color=(1,0,0), clip_on=False)

plt.xlabel('S_t (uM)')
plt.ylabel('k_EDS (1/hour)')
plt.legend()
plt.gca().set_yscale('linear')
plt.gca().set_xscale('linear')
# plt.gca().set_xlim([-2, 40])
# plt.gca().set_ylim([-0.1, 4])
plt.savefig(fig2_name, format="svg")
plt.show()


