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
This script extracts ternary complex assembly and activity parameters
from experimental k_E3 rates previously saved by a
..._get_rates.py script.
Fitting is performed using a system of equations that
describe binding equilibria of molecular glues.
"""

# ### =============================================================================
# ### SECTION 1: Initialization
# ### =============================================================================

# ### --- Import libraries and commands ---

import numpy as np
from scipy.optimize import least_squares, fsolve
import matplotlib.pyplot as plt
import scipy.io

### --- Set parameters ---

# BioTek Synergy plate readers read a rectangular area of the plate.
# We typically set up experiments so that D_t (degrader) titrations run column-wise:
#   • Within a column (e.g., A1, B1, C1, …), wells have different D_t concentrations.
#   • Within a row across columns (e.g., A1, A2, A3, …), wells have the same D_t concentration.
# Different columns may use different E_t (enzyme) and S_t (substrate) concentrations,
# while all rows within a column share the same E_t and S_t.

# MAT file with experimental k_E3 rates (2D array):
#   k_E3.shape[0] -> number of measured rows
#   k_E3.shape[1] -> number of measured columns
mat_file_name="Poma_SALL4_aa405_432_exp01_fit2_3par_results.mat"

# Select rows/columns (0-based indices) to include in this fit.
# Note: the upper-left well of the measured area corresponds to k_E3[0, 0].
# The length of row_selection must equal D_t.size.
# The length of col_selection must equal E_t.size and S_t.size.
row_selection=[0,2,3,4,5,6,7,8,9,10,11]
col_selection=[0,1,2,3,4,5,6]

# D_t concentrations (nM) for the selected rows, in the order given by row_selection.
D_t = np.array([ 0, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2])

# E_t and S_t concentrations (nM) for the selected columns, in the order given by col_selection.
E_t = np.array([ 40, 40, 40, 40, 40, 40, 40])
S_t = np.array([ 0, 732, 1470, 3700, 7300, 14700, 29300])

# Initial guesses for fitted parameters (units):
#   K_ED, K_ES, K_EDS in nM
#   k_cat_over_K_m in (1 / (mM·s)); for dBET1, BRD4 and cereblon system k_cat_over_K_m ~ 30 (1 / (mM·s))
K_ED = 100
K_ES = 10000
K_EDS = 10000
k_cat_over_K_m = 10

# If there is a systematic bias in nominal total concentrations (E_t, S_t, D_t),
# apply a correction coefficient here (1.0 = no correction).
# In high-quality, saturating datasets, this factor can be floated in least_squares().
coef_1=1

# See Section 3 for background_rate definition and subtraction:
#   k_tot = k_E3 - background_rate

# Output file names (change suffixes if desired).
# fig1: plot of raw k_E3 values for the selected samples
fig1_name=mat_file_name.split("_fit")[0]+"_k_E3_plot.svg"

# fig2: background-corrected rates (k_tot) vs D_t
fig2_name=mat_file_name.split("_fit")[0]+"_k_tot_vs_D_t_fit_result.svg"

# fig3: background-corrected rates (k_tot) vs S_t
fig3_name=mat_file_name.split("_fit")[0]+"_k_tot_vs_S_t_fit_result.svg"

# txt: fitted parameter values
txt1_name=mat_file_name.split("_fit")[0]+"_fitted_parameters.txt"

# (Optional) Basic sanity checks—uncomment if helpful:
# assert D_t.size == len(row_selection), "D_t length must match row_selection."
# assert E_t.size == len(col_selection) and S_t.size == len(col_selection), \
#        "E_t and S_t lengths must match col_selection."
# assert np.all(np.array(row_selection) >= 0) and np.all(np.array(col_selection) >= 0), \
#        "Selections must be non-negative indices."
# assert np.all(np.array(row_selection) < scipy.io.loadmat(mat_file_name)['k_E3'].shape[0]), \
#        "Selections indices exceed the total number of rows"
# assert np.all(np.array(col_selection) < scipy.io.loadmat(mat_file_name)['k_E3'].shape[1]), \
#        "Selections indices exceed the total number of columns"


# ### =============================================================================
# ### SECTION 2: Function Definitions
# ### =============================================================================

# ----- Compute [EDS] and [ES] and calculate the catalytic rate k_tot = k_cat_over_K_m * ([EDS] + [ES])

# This is the main function of the script.
# It numerically solves for the concentrations of three unknowns—[EDS], [ES], and free [S]—
# given total concentrations E_t, S_t, and D_t, and parameter values K_ED, K_ES, K_EDS, and k_cat_over_K_m.
# The solution is obtained by solving the system of mass conservation equations
# that describe molecular glue binding equilibria (see TPD_glue_system).
# The vector of unknowns is represented by array x: x[0] = [EDS], x[1] = [ES], x[2] = [S].

def rate_vs_D_t_fsolve_glue(K_ED, K_ES, K_EDS, E_t, S_t, D_t, k_cat_over_K_m, coef_1):
    kcat_Km_per_nM_per_h = k_cat_over_K_m * 3600 / 1_000_000  # Convert rate constant to per nM per hour
    rate = np.zeros((D_t.size, S_t.size))  # Initialize array for calculated rates with the same shape as k_E3
    tol = 0.01  # Tolerance for numerical error to allow small boundary violations
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
                fun = lambda x: TPD_glue_system(x, K_ED, K_ES, K_EDS, E_t[j], S_t[j], D_t[i], coef_1)
                # Solve for the root using fsolve
                x, _, ier, _ = fsolve(fun, x0, full_output=True)
                # Define error tolerance for root validation
                error = tol * min([l for l in [E_t[j], S_t[j] * coef_1, D_t[i]] if l != 0])

                # Check whether the root is within acceptable bounds
                if not ((x[0] <= min(E_t[j], S_t[j] * coef_1, D_t[i]) + error) and (x[0] >= -error) and
                       (x[1] <= min(E_t[j], S_t[j] * coef_1) + error) and (x[1] >= -error) and
                       (x[2] <= S_t[j] * coef_1 + error) and (x[2] >= np.maximum(S_t[j] - E_t[j],0)-error)):
                    print(f'[{i},{j}]: invalid root, retrying with new initial guess') # comment out to silence
                    x0=initial_guess(E_t[j],S_t[j],D_t[i])
                    continue
                
                # Retry with a new guess if fsolve exceeded its iteration limit
                if ier != 1:
                    print(f'[{i},{j}]: fsolve() did not converge, retrying with new initial guess') # comment out to silence
                    x0=initial_guess(E_t[j],S_t[j],D_t[i])
                    continue
                
                # Calculate the catalytic rate k_tot
                rate[i, j] = kcat_Km_per_nM_per_h * (x[0] + x[1])
                x0=x
                break
            
    return rate


# ----- System of three mass conservation equations for molecular glue binding equilibria

def TPD_glue_system(x, K_ED, K_ES, K_EDS, E_t, S_t, D_t, coef_1):
    # Initialize the output array F
    F = np.zeros(3)

    # The vector of unknowns is x: x[0] = [EDS], x[1] = [ES], x[2] = free [S]

    # Define the system of equations based on mass conservation and equilibrium binding
    F[0] = x[1] * K_ES + x[0] * K_EDS + x[1] * x[2] + x[0] * x[2] - E_t * x[2]
    F[1] = x[0] * x[2] * ((K_EDS * K_ED) / K_ES) + x[0] * x[1] * K_EDS + x[0] * x[1] * x[2] - D_t * x[1] * x[2]
    F[2] = x[0] + x[1] + x[2] - S_t * coef_1

    return F

def initial_guess(E_t, S_t, D_t):
    # x0[0]: random value between 0 and min(E_t, S_t, D_t)
    x0_0 = np.random.rand() * min(E_t, S_t, D_t)
    
    # x0[1]: random value between 0 and min(E_t, S_t)
    x0_1 = np.random.rand() * min(E_t, S_t)

    # x0[2]: random value between max(S_t - E_t, 0) and S_t
    lower_bound_2 = np.maximum(S_t - E_t, 0)
    upper_bound_2 = S_t
    x0_2 = np.random.rand() * (upper_bound_2 - lower_bound_2) + lower_bound_2

    return np.array([x0_0, x0_1, x0_2])

# # # ------ Definitions of color arrays for the plots

# colors_rgb = [
#     (0.96, 0.043, 0.043),
#     (0.98039216, 0.88235294, 0.05490196),
#     (0.0627451 , 0.62352941, 0.34509804),
#     (0.40784314, 0.08627451, 0.61960784),
#     (1, 0.0, 1),
#     (0.96, 0.043, 0.043),
#     (0.98039216, 0.88235294, 0.05490196),
#     (0.0627451 , 0.62352941, 0.34509804),
#     (0.40784314, 0.08627451, 0.61960784),
#     (1, 0.0, 1)
# ]

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
# ### SECTION 3: Load k_E3 rates from mat_file_name, visualize them, and subtract background rate
# ### =============================================================================

# Load the k_E3 rate matrix from the specified .mat file
mat_file_data = scipy.io.loadmat(mat_file_name)
rate_buffer = mat_file_data['k_E3']

# Select the subset of rows and columns for parameter fitting
rate_selection = rate_buffer[np.ix_(row_selection, col_selection)]

# ----------- Plot raw k_E3 rates without background subtraction -----------
# ----------- Comment out this block if the plot is not needed -------------

plt.figure(1, figsize=(6, 6))
plt.clf()

for i in range(S_t.size):
    plt.scatter(D_t[:], rate_selection[:, i], color=colors_rgb[i], facecolors='none',
                label=f'S_t = {S_t[i]} nM', clip_on=False)

plt.xlabel('D_t (nM)')
plt.ylabel('k_E3 (1/hour)')
# plt.title('')
plt.legend()
plt.gca().set_yscale('linear')
plt.gca().set_xscale('log')
plt.gca().set_xlim([1, 10000])
# plt.gca().set_ylim([-0.1, 4])
plt.savefig(fig1_name, format="svg")
# plt.show()

# ----------- End of raw rate plotting (Figure 1) -----------

# ---------- Subtract background rate ----------
# For molecular glue systems, background rate can be estimated as
# the average of all k_E3 rates in the column corresponding to S_t == 0

indices = np.where(S_t == 0)[0]
if indices.size > 0:
    S_t_zero_index = indices[0]
else:
    print("No zero found in S_t. Background rate was not subtracted")

background_rate = np.mean(rate_selection[:, S_t_zero_index])

# Final rate data used for fitting: background-subtracted k_E3 values
rate_data_values = rate_selection - background_rate

# ### =============================================================================
# ### SECTION 4: Estimate parameter values using least_squares() fitting
# ### =============================================================================

# Assign initial guesses and lower/upper bounds for the fitted parameters
z0 = [K_ED, K_ES, K_EDS, k_cat_over_K_m]
lb = [0.01, 0.01, 0.01, 0]
ub = [1000000, 1000000, 1000000, 10000]

# Define the residuals function for least_squares()
def rate_residuals_fun(z):
    return np.subtract(
        rate_data_values,
        rate_vs_D_t_fsolve_glue(z[0], z[1], z[2], E_t, S_t, D_t, z[3], coef_1)
    ).flatten()

# Perform nonlinear least-squares fitting
result = least_squares(rate_residuals_fun, z0, bounds=(lb, ub))

# Extract best-fit parameter values from the result
K_ED = result.x[0]
K_ES = result.x[1]
K_EDS = result.x[2]
k_cat_over_K_m = result.x[3]

# Estimate standard errors of fitted parameters
# result.jac contains the Jacobian matrix; result.fun contains residuals

J = result.jac
residuals = result.fun

# Degrees of freedom = number of data points – number of fitted parameters
degrees_of_freedom = len(rate_residuals_fun(result.x)) - len(result.x)

# Residual variance (chi-squared estimate)
residual_variance = np.sum(residuals**2) / degrees_of_freedom

# Covariance matrix of the fitted parameters
cov_matrix = np.linalg.inv(J.T.dot(J)) * residual_variance

# Standard errors = square root of diagonal elements of covariance matrix
standard_errors = np.sqrt(np.diag(cov_matrix))

# ### =============================================================================
# ### SECTION 5: Plot fitting results and output parameter values
# ### =============================================================================

# ------- Plot experimental and fitted k_tot vs D_t for different S_t concentrations

plt.figure(2, figsize=(6, 6))
plt.clf()

# Define D_t range for plotting the fitted curves
D_t_range = np.logspace(np.log10(np.min(D_t[D_t != 0])) - 0.2, np.log10(np.max(D_t[D_t != 0])) + 0.2, 100)
# Compute theoretical curves for all S_t values
fitted_curves = rate_vs_D_t_fsolve_glue(K_ED, K_ES, K_EDS, E_t, S_t, D_t_range, k_cat_over_K_m, coef_1)

# Plot experimental points and corresponding fitted curves
for i in range(S_t.size):
    plt.scatter(D_t[:], rate_data_values[:, i], color=colors_rgb[i], facecolors='none', label=f'S_t = {S_t[i]/1000} uM', clip_on=False)
    plt.plot(D_t_range, fitted_curves[:, i], '-', color=colors_rgb[i], clip_on=False)

plt.xlabel('D_t (nM)')
plt.ylabel('k_tot (1/hour)')
plt.legend()
plt.gca().set_yscale('linear')
plt.gca().set_xscale('log')
# plt.gca().set_xlim([1, 10000])
# plt.gca().set_ylim([-0.1, 4])
plt.savefig(fig2_name, format="svg")
# plt.show()


# ------- Plot k_tot vs S_t for minimal and maximal D_t values
# ------- This helps visualize how much the molecular glue enhances E3–substrate affinity

plt.figure(3, figsize=(6, 6))
plt.clf()

# Define S_t range for plotting
S_t_range = np.linspace(0, S_t.max() * 1.1, 50)

# Generate E_t array of same size as S_t_range (assumes constant E_t across all S_t)
E_t_range = np.linspace(E_t[0], E_t[0], 50)

# Compute fitted curves for lowest and highest D_t values
min_max_D_t_rate_fits = rate_vs_D_t_fsolve_glue(K_ED, K_ES, K_EDS, E_t_range, S_t_range, np.array([D_t.min(), D_t.max()]), k_cat_over_K_m, coef_1)

# Plot experimental data and fitted curves for min and max D_t
plt.scatter(S_t[:]/1000, rate_data_values[np.argmin(D_t),:], color=(0,0,0), facecolors='none', label=f'D_t = {np.min(D_t)} nM', clip_on=False)
plt.plot(S_t_range/1000, min_max_D_t_rate_fits[0,:], '-', color=(0,0,0), clip_on=False)
plt.scatter(S_t[:]/1000, rate_data_values[np.argmax(D_t),:], color=(1,0,0), facecolors='none', label=f'D_t = {np.max(D_t)} nM', clip_on=False)
plt.plot(S_t_range/1000, min_max_D_t_rate_fits[1,:], '-', color=(1,0,0), clip_on=False)

plt.xlabel('S_t (uM)')
plt.ylabel('k_tot (1/hour)')
plt.legend()
plt.gca().set_yscale('linear')
plt.gca().set_xscale('linear')
# plt.gca().set_xlim([-2, 40])
# plt.gca().set_ylim([-0.1, 4])
plt.savefig(fig3_name, format="svg")

# ------- Output fitted parameter values to the screen and a text file

param_names = ["K_ED", "K_ES", "K_EDS", "k_cat_over_Km"]

# Print parameter values with standard errors to the console
for i in range(len(param_names)):
    print(f"{param_names[i]} = {round(result.x[i],3)} ± {round(standard_errors[i],3)}")

# Save parameter values with standard errors to a text file
with open(txt1_name, "w") as f:
    for i in range(len(param_names)):
        print(f"{param_names[i]} = {round(result.x[i],3)} ± {round(standard_errors[i],3)}", file=f)


plt.show()

