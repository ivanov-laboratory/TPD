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
Extracts catalytic k_E3 rates from experimental FRET decay curves. The script
parses raw donor and acceptor fluorescence intensity data, calculates FRET
(acceptor/donor), and fits the resulting FRET decay curves to a time-dependent
inactivation (TDI) model to obtain kinetic parameters.

Kinetic parameters are first extracted using a 5-parameter fit (Fit 1)
performed on a selection of samples from a longer-acquisition experiment
(file_name_1). In Fit 1, three parameters (k_E3, F_max, and k_inact) are fitted
individually for each sample, and two parameters (k_0 and F_min) are fitted
globally for the entire selection. A long acquisition (4–6 hours) is required
for reliable fitting of all five free parameters of the TDI model. The
5-parameter Fit 1 can be computationally expensive and therefore works best on
a smaller selection of samples. When applied to the entire dataset (optional
SECTION 6), the fitting is performed one column (i.e., one D_t titration
series) at a time. The globally fitted parameters extracted from a selection
of samples are saved in a .mat matrix file and can be used for the analysis of
other, similar datasets (see Fit 2 below).

Fit 2 uses the parameter values saved by Fit 1 to perform a 2-parameter,
2-plus-1-parameter, or 3-parameter fit on a dataset (file_name_2). The most
appropriate fitting strategy should be selected by uncommenting one of the
three options in SECTION 7 of the script. Fit 2 can be performed on
shorter-acquisition datasets (1–2 hours) and on datasets containing a large
number of samples. Fit 2 can also be performed on the same dataset as Fit 1
(file_name_1) by selecting only the initial 1–2 hours of the dataset.

Values of the kinetic parameters extracted using Fit 1 or Fit 2 are saved as
2D arrays in a .mat file for further analysis by scripts such as
..._glue_fit_rates.py or ..._protac_fit_rates.py.
"""

### =============================================================================
### SECTION 1: Initialization
### =============================================================================

### --- Import libraries and commands ---
import sys
import os
import numpy as np
import scipy.io
from io import StringIO
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.io import savemat

### --- Set parameters ---

# # # Specify names of text files containing FRET data exported by the plate reader.
# # # This script can analyze one or two experiments:
# # #   • A longer acquisition (file_name_1) can be used to determine all five parameters of the TDI model (Fit 1).
# # #   • These parameters can then be reused to extract k_E3 from a shorter acquisition (file_name_2) 
# # #     by performing a 2-, 2plus1-, or 3-parameter fit (Fit 2).
file_name_1 = "dBET1_BRD4_exp03.txt"
file_name_2 = "dBET1_BRD4_exp03.txt"  # file_name_2 can be the same as file_name_1

# ----------------------------------------------------------------------
# How the experiment is set up:
#
# - The microplate is arranged with PROTAC (degrader) titration series in columns.
#   Example: wells A1, B1, C1, D1, ... are one titration series with increasing degrader concentration D_t.
#
# - Wells in the same row (e.g., A1, A2, A3, A4, ...) usually contain the same degrader concentration,
#   but different combinations of enzyme (E_t) and substrate (S_t) concentrations.
#
# - All wells in the same column share the same E_t and S_t concentrations,
#   but different columns can have different E_t and S_t.
#
# The BioTek Synergy Neo2 exports kinetic readouts from a rectangular block of wells.
# In the output file:
#   - Each line corresponds to a single time point in the kinetic run.
#   - The readings for all measured wells are written into a single line row-by-row, left to right.
#     Example: A1, A2, A3, ..., B1, B2, B3, ...
#
# IMPORTANT: Python uses zero-based indexing!
# For example, well A1 is row_index=0, col_index=0.
# Well B1 is row_index=1, col_index=0, and so on.
# ----------------------------------------------------------------------

# # # Specify how many D_t titration series were performed (i.e., how many columns were measured)
# # # These values correspond to the total number of columns in the rectangular area of the plate that was read.
# # # They can differ between file_name_1 and file_name_2.
D_t_series_total_1 = 8
D_t_series_total_2 = 8

# # # Select rows and columns in file_name_1 to be used in Fit 1 (full 5-parameter fit).
# # # It is recommended not to select more than ~40 wells total, to keep the fit computationally efficient.
row_selection_fit_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
col_selection_fit_1 = [1,2,3]

# # # Select rows and columns in file_name_2 for which the results of Fit 2 will be plotted.
# # # Note: Fit 2 is performed on the entire dataset; this selection is only used for plotting.
row_selection_fit_2_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
col_selection_fit_2_plot = [3]

# # # Specify timepoint indices to include in Fit 1.
# # # Use None to include all time points.
index_start_fit1 = None
index_end_fit1 = None

# # # Specify timepoint indices to include in Fit 2.
# # # Use None to include all time points.
index_start_fit2 = 0
index_end_fit2 = 120

# # # Generate output file names from input file names.
# # # Edit only if a different name extension is desired.
exp1_name = file_name_1.split(".txt")[0]
exp2_name = file_name_2.split(".txt")[0]

# # # fig_1: plot of the results of Fit 1
fig_1_name = exp1_name + "_fit1_plot.svg"

# # # fig_2: plot of k_inact vs k_E3 for wells from Fit 1
fig_2_name = exp1_name + "_k_inact_vs_k_E3_plot.svg"

# # # fig_3: plot of the results of Fit 2
fig_3_name = exp2_name + "_fit2_plot.svg"

# # # .mat file containing global parameters from Fit 1 on selected wells from file_name_1 (defined by row_selection_fit_1 and col_selection_fit_1)
mat_global_params = exp1_name + "_fit1sel_global_params.mat"

# # # .mat file containing results of 5-parameter Fit 1 for the entire file_name_1
mat_fit1_results = exp2_name + "_fit1_5par_results.mat"

# # # .mat files containing results of the selected Fit 2 (see SECTION 7) for the entire file_name_2
mat_fit2_results_2par = exp2_name + "_fit2_2par_results.mat"
mat_fit2_results_2plus1par = exp2_name + "_fit2_2plus1par_results.mat"
mat_fit2_results_3par = exp2_name + "_fit2_3par_results.mat"


### =============================================================================
### SECTION 2: Function Definitions
### =============================================================================

def split_text_into_sections(file_path):
    # # # Read a text file and split it into sections separated by two or more newlines.
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Filter out comment lines and empty lines
    filtered_lines = [line for line in lines if not line.strip().startswith('#')]

    # Join and split into sections
    text = ''.join(filtered_lines)
    sections = text.strip().split('\n\n')
    
    return sections

def time_to_hours(time_str):
    # # # Convert time in HH:MM:SS format to hours (float)
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600

def load_data_from_string(data_string):
    # # # Load data from a tab-delimited string while:
    # # #   • Converting time column (index 0) to hours
    # # #   • Skipping temperature column (index 1)
    # # #   • Removing malformed rows

    data_file = StringIO(data_string)
    first_line = data_file.readline()
    num_columns = len(first_line.strip().split('\t'))
    data_file.seek(0)

    # Define column indices to keep (skip index 1)
    usecols = tuple(i for i in range(num_columns) if i != 1)

    # Collect only well-formed lines
    valid_lines = []
    for line in data_file:
        columns = line.strip().split('\t')
        if len(columns) != num_columns:
            print(f"Stopping read: Line has unexpected format:\n{line}")
            break
        valid_lines.append(line)

    valid_data_string = StringIO("\n".join(valid_lines))
    converters = {0: time_to_hours}

    # Load numerical data
    data = np.loadtxt(valid_data_string, delimiter='\t', usecols=usecols, converters=converters)
    
    return data

def FRET_decay(hours, k_E3, F_max, k_inact, k_0, F_min):
    # # # Solve the time-dependent inactivation (TDI) model
    # # # Returns theoretical FRET decay curve for given kinetic parameters

    def dData_dt(t, y, k_E3, k_0, k_inact):
        dydt = -1 * (y - F_min) * (k_E3 * np.exp(-k_inact * t) + k_0)
        return dydt

    sol = solve_ivp(dData_dt, [hours[0], hours[-1]], [F_max], args=(k_E3, k_0, k_inact), t_eval=hours)
    return sol.y[0]

def FRET_decay_resid_5par(x, FRET_sel, hours, index_start, index_end):
    # # # Return residuals between experimental FRET decays and the 5-parameter model.
    # # # Fit: one (k_E3, F_max, k_inact) triplet per well; k_0 and F_min fitted globally.

    pred_values = np.zeros_like(FRET_sel[index_start:index_end, :])
    for i in range(FRET_sel.shape[1]):
        pred_values[:, i] = FRET_decay(
            hours[index_start:index_end],
            x[3*i], x[3*i+1], x[3*i+2],
            x[FRET_sel.shape[1]*3], x[FRET_sel.shape[1]*3+1]
        )
    return (FRET_sel[index_start:index_end] - pred_values).flatten()

def FRET_decay_resid_2par(x, FRET, hours, index_start, index_end, k_inact, k_0, F_min):
    # # # Return residuals for 2-parameter fit: [k_E3, F_max]
    # # # Assumes fixed k_inact, k_0, and F_min

    pred = FRET_decay(hours[index_start:index_end], x[0], x[1], k_inact, k_0, F_min)
    return (FRET[index_start:index_end] - pred).flatten()

def FRET_decay_resid_2plus1_par(x, FRET, hours, index_start, index_end, k_0, F_min, slope, intercept):
    # # # Return residuals for 2plus1-parameter fit: [k_E3, F_max]
    # # # k_inact is computed from k_E3 using: k_inact = slope * k_E3 + intercept

    k_E3 = x[0]
    k_inact = k_E3 * slope + intercept
    pred = FRET_decay(hours[index_start:index_end], k_E3, x[1], k_inact, k_0, F_min)
    return (FRET[index_start:index_end] - pred).flatten()

def FRET_decay_resid_3par(x, FRET, hours, index_start, index_end, k_0, F_min):
    # # # Return residuals for 3-parameter fit: [k_E3, F_max, k_inact]
    # # # Assumes fixed k_0 and F_min

    pred = FRET_decay(hours[index_start:index_end], x[0], x[1], x[2], k_0, F_min)
    return (FRET[index_start:index_end] - pred).flatten()

# # # Define RGB color list for plotting
colors_rgb = [
    (0, 0, 0),    
    (0.96, 0.043, 0.043),
    (0.98, 0.62, 0.055),
    (0.98039216, 0.88235294, 0.05490196),
    (0.77254902, 0.69803922, 0.07058824),
    (0.54901961, 0.83921569, 0.09411765),
    (0.11372549, 1.0, 0.55686275),
    (0.0627451, 0.62352941, 0.34509804),
    (0.04705882, 0.6, 0.69803922),
    (0.0745098, 0.85882353, 1.0),
    (0.16078431, 0.10588235, 1.0),
    (0.40784314, 0.08627451, 0.61960784)
]

### =============================================================================
### SECTION 3: Perform 5-parameter fit (Fit 1) on a selection of wells from file_1
### =============================================================================

# # # Open the FRET data file and split into sections
sections = split_text_into_sections(file_name_1)

# # # Load donor channel data (e.g., fluorescein, 528 nm) from section 1
data_528 = load_data_from_string(sections[1])

# # # Load acceptor channel data (e.g., AF594, 635 nm) from section 3
data_635 = load_data_from_string(sections[3])

# # # Extract time points (in hours) from the first column
hours = data_528[:, 0]

# # # Calculate FRET efficiency: acceptor / donor (omit time column)
FRET = np.divide(data_635[:, 1:], data_528[:, 1:])

# # # Select wells for Fit 1 using row_selection_fit_1 and col_selection_fit_1
exp_selection = []
for i in col_selection_fit_1:
    for j in row_selection_fit_1:
        exp_selection.append(i + j * D_t_series_total_1)

FRET_sel = FRET[:, exp_selection]

print(f"Performing 5-par fit on file {file_name_1} using the following well selection:")
print(f"column(s) = {col_selection_fit_1}")
print(f"row(s) = {row_selection_fit_1}")

# # # Initialize arrays for per-well parameters
k_E3 = np.zeros(FRET_sel.shape[1])
F_max = np.zeros(FRET_sel.shape[1])
k_inact = np.zeros(FRET_sel.shape[1])

# # # Initialize parameter arrays for least_squares()
# # # Three parameters are fitted per well: k_E3, F_max, and k_inact
# # # Two parameters are fitted globally: k_0 and F_min
x = np.zeros(FRET_sel.shape[1]*3 + 2)
x0 = np.ones_like(x)
lb = np.zeros_like(x)
ub = np.zeros_like(x)

# # # Assign initial parameter guesses for least_squares()
x0[0:FRET_sel.shape[1]*3:3] = 1
x0[1:FRET_sel.shape[1]*3:3] = 1
x0[2:FRET_sel.shape[1]*3:3] = 1
x0[FRET_sel.shape[1]*3] = 0.05     # initial guess for k_0
x0[FRET_sel.shape[1]*3+1] = 0.08   # initial guess for F_min

# # # Define residual function and perform least_squares minimization
fun_FRET_ls = lambda x: FRET_decay_resid_5par(x, FRET_sel, hours, index_start_fit1, index_end_fit1)
result = least_squares(fun_FRET_ls, x0)

# # # ----- Extract and print fitted parameters -----

# print(f"5-par fit results:") # comment out if not needed

# # # k_E3 values
k_E3[:] = result.x[0:FRET_sel.shape[1]*3:3]
# print(f"k_E3 = {np.round(k_E3[:], 3)}") # comment out if not needed

# # # F_max values
F_max[:] = result.x[1:FRET_sel.shape[1]*3:3]
# print(f"F_max = {np.round(F_max[:], 3)}") # comment out if not needed

# # # k_inact values
k_inact[:] = result.x[2:FRET_sel.shape[1]*3:3]
# print(f"k_inact = {np.round(k_inact[:], 3)}") # comment out if not needed

# # # Extract global parameters k_0 and F_min for use in Fit 2
k_0 = result.x[FRET_sel.shape[1]*3]
print(f"k_0 = {round(k_0, 3)}", end=' ')

F_min = result.x[FRET_sel.shape[1]*3+1]
print(f"F_min = {round(F_min, 3)}")

# # # Calculate mean of k_inact values for use in Fit 2
k_inact_mean = np.mean(k_inact[:])
print(f"k_inact_mean = {round(k_inact_mean, 3)}", end=' ')

### =============================================================================
### SECTION 4: Generate a plot showing results of the 5-parameter fit (Fit 1) for selected wells 
### =============================================================================

plt.figure(1, figsize=(6, 6))

fit_index = 0
for i in col_selection_fit_1:
    color_index = 0  # reset color sequence for each column
    for j in row_selection_fit_1:
        sample_index = i + j * D_t_series_total_1
        color = colors_rgb[color_index % len(colors_rgb)]
        color_index += 1

        # plot experimental FRET values
        plt.scatter(hours, FRET[:, sample_index], s=5, color=color, facecolors='none', linewidths=0.5, label=f'Column {i+1}', clip_on=False)

        # plot best-fit FRET decay curve using fitted parameters
        plt.plot(
            hours[:],
            FRET_decay(
                hours[:],
                result.x[3*fit_index],
                result.x[3*fit_index+1],
                result.x[3*fit_index+2],
                result.x[FRET_sel.shape[1]*3],
                result.x[FRET_sel.shape[1]*3+1]
            ),
            '-', color=color, linewidth=2, clip_on=False
        )
        fit_index += 1

plt.xlabel('time (hours)')
plt.ylabel('FRET (a.u.)')
plt.title('Fit 1: 5-parameter fit for selected wells')

# # Uncomment below to enable interactive viewing or axis limits
# plt.legend()
# plt.grid(True)
# plt.gca().set_xlim([0, 2])
# plt.gca().set_ylim([0.2, 0.75])

# # Save the plot to a vector graphic file for figure generation
plt.savefig(fig_1_name, format="svg")

# # Or output to screen for interactive viewing
# plt.show()


### =============================================================================
### SECTION 5: Investigate the relationship between k_inact and k_E3 and save globally fitted parameter values into a .mat file
### =============================================================================


# # In this section a potential linear dependence between k_inact and k_E3 is investigated
# # and slope and intercept parameters are calculated for such linear dependence
# # these parameters are saved into a Fit 1 global parameters file.
# # The slope and intercept values are used in the 2plus1-parameter option in Fit 2 (see SECTION 7)

# # If a linear relationship exists, we can express k_inact as:
# # k_inact = k_E3 * slope + intercept

# # Perform a least-squares fit of k_inact vs k_E3
x0 = np.ones(2)  # initial guesses for slope and intercept
fun_k_inact_vs_k_E3_ls = lambda x: (k_inact[:] - (k_E3[:] * x[0] + x[1]))
k_inact_ls_result = least_squares(fun_k_inact_vs_k_E3_ls, x0)

k_inact_vs_k_E3_slope = k_inact_ls_result.x[0]
k_inact_vs_k_E3_intercept = k_inact_ls_result.x[1]

print(f"k_inact_vs_k_E3_slope = {round(k_inact_vs_k_E3_slope, 3)}", end=' ')
print(f"k_inact_vs_k_E3_intercept = {round(k_inact_vs_k_E3_intercept, 3)}")

# # Generate a plot of k_inact vs k_E3 and overlay linear fit
plt.figure(2, figsize=(6, 6))

fit_index = 0
for i in col_selection_fit_1:
    color_index = 0
    for j in row_selection_fit_1:
        sample_index = i + j * D_t_series_total_1
        color = colors_rgb[color_index % len(colors_rgb)]
        color_index += 1
        plt.scatter(k_E3[fit_index], k_inact[fit_index], s=100, color=color, facecolors=color, linewidths=0.5, clip_on=False)
        fit_index += 1

# Plot best-fit line
x_fit = np.array([np.min(k_E3), np.max(k_E3)])
y_fit = x_fit * k_inact_vs_k_E3_slope + k_inact_vs_k_E3_intercept
plt.plot(x_fit, y_fit, '-', color=colors_rgb[0], linewidth=2, clip_on=False)

plt.xlabel('k_E3')
plt.ylabel('k_inact')
plt.title('k_inact vs. k_E3')

# # Save the plot to a vector graphic file
plt.savefig(fig_2_name, format="svg")

# # Or show interactively (execution will pause until the window is closed)
# plt.show()

# # Save globally fitted parameters for later use in Fit 2
# # If this file already exists and is acceptable, you can comment out Sections 3–6 to speed up execution
savemat(
    mat_global_params,
    {
        'k_inact_mean': k_inact_mean,
        'k_0_fit1sel': k_0,
        'F_min_fit1sel': F_min,
        'k_inact_vs_k_E3_slope': k_inact_vs_k_E3_slope,
        'k_inact_vs_k_E3_intercept': k_inact_vs_k_E3_intercept
    }
)

### =============================================================== 
### SECTION 6: Perform 5-parameter fit on the entire file_name_1 dataset one column (D-t titration series) at a time
### ================================================================

### In this section 5-parameter fit is performed on the entire file_name_1 dataset one column at a time 
### and the results are saved into a .mat file (..._fit1_5par_results.mat).
### That matrix file can be used as input by ..._glue_fit_rates.py or ..._protac_fit_rates.py scripts.
### If SECTION 6 is executed, Fit 2 becomes optional and SECTIONS 7 and 8 can be commented out from this script.
### Alternatively SECTION 6 can be commented out if 2-, 2plus1- or 3-parameter Fit 2 is preferred.

k_E3 = np.zeros((int(FRET.shape[1] / D_t_series_total_1), D_t_series_total_1))
F_max = np.zeros_like(k_E3)
k_inact = np.zeros_like(k_E3)
k_0 = np.zeros_like(k_E3)
F_min = np.zeros_like(k_E3)

print(f"Performing 5-par fit on the entire file {file_name_1}")

for j in range(D_t_series_total_1):

    print(f"\rFitting column {j+1}", end="", flush=True)

    indices_selection=np.arange(j, FRET.shape[1], D_t_series_total_1)
    FRET_sel=FRET[:, indices_selection]

    x = np.zeros(FRET_sel.shape[1]*3 + 2)
    x0 = np.ones_like(x)
    lb = np.zeros_like(x)
    ub = np.zeros_like(x)

    # # # Assign initial parameter guesses for least_squares()
    x0[0:FRET_sel.shape[1]*3:3] = 1
    x0[1:FRET_sel.shape[1]*3:3] = 1
    x0[2:FRET_sel.shape[1]*3:3] = 1
    x0[FRET_sel.shape[1]*3] = 0.05     # initial guess for k_0
    x0[FRET_sel.shape[1]*3+1] = 0.08   # initial guess for F_min

    # # # Define residual function and perform least_squares minimization
    fun_FRET_ls = lambda x: FRET_decay_resid_5par(x, FRET_sel, hours, index_start_fit1, index_end_fit1)
    result = least_squares(fun_FRET_ls, x0)

    k_E3[:,j] = result.x[0:FRET_sel.shape[1]*3:3]
    F_max[:,j] = result.x[1:FRET_sel.shape[1]*3:3]
    k_inact[:,j] = result.x[2:FRET_sel.shape[1]*3:3]
    k_0[:,j] = result.x[FRET_sel.shape[1]*3]
    F_min[:,j] = result.x[FRET_sel.shape[1]*3+1]

print(f"\n")

savemat(
    mat_fit1_results,
    {'k_E3': k_E3, 'F_max': F_max, 'k_inact': k_inact, 'k_0': k_0, 'F_min': F_min}
)


### =============================================================================
### SECTION 7: Perform 2-parameter, 2plus1-parameter, or 3-parameter fit (Fit 2)
### on the entire dataset in file_name_2 using globally fitted parameter values from Fit 1.
### Uncomment the desired option of the three options below.
### =============================================================================

# Open file and split into sections using double newline as delimiter
sections = split_text_into_sections(file_name_2)

# Load donor (fluorescein) and acceptor (AF594) fluorescence intensities
data_528 = load_data_from_string(sections[1])
data_635 = load_data_from_string(sections[3])

# Extract time values (in hours) from first column
hours = data_528[:, 0]

# Calculate FRET signal by dividing acceptor by donor (excluding time column)
FRET = np.divide(data_635[:, 1:], data_528[:, 1:])

# Load globally fitted parameters from Fit 1 on file_name_1 selection defined by row_selection_fit_1 and col_selection_fit_1
fit1sel_global_params = scipy.io.loadmat(mat_global_params)
k_inact_mean = fit1sel_global_params['k_inact_mean'].item()
k_0_fit1sel = fit1sel_global_params['k_0_fit1sel'].item()
F_min_fit1sel = fit1sel_global_params['F_min_fit1sel'].item()
k_inact_vs_k_E3_slope = fit1sel_global_params['k_inact_vs_k_E3_slope'].item()
k_inact_vs_k_E3_intercept = fit1sel_global_params['k_inact_vs_k_E3_intercept'].item()

# # # ======================= Option 1: 2-parameter Fit =======================
# # # in this option, k_inact_mean value from Fit 1 is used for all samples in file_name_2


print(f"Performing 2-par fit on file {file_name_2} using time points from {hours[index_start_fit2]} to {hours[index_end_fit2]} hours")
print("The following parameters are fixed:")
print(f"k_inact = {np.round(k_inact_mean, 3)}; k_0 = {np.round(k_0_fit1sel, 3)}; F_min = {np.round(F_min_fit1sel, 3)}")

plot_title = "Fit 2: 2-parameter fit"

# Initialize output arrays
k_E3 = np.zeros((int(FRET.shape[1] / D_t_series_total_2), D_t_series_total_2))
F_max = np.zeros_like(k_E3)
k_inact = np.zeros_like(k_E3)
k_0 = np.zeros_like(k_E3)
F_min = np.zeros_like(k_E3)


# Fit each FRET decay individually
x0 = np.ones(2)
for l in range(D_t_series_total_2):
    for m in range(int(FRET.shape[1] / D_t_series_total_2)):
        print(f"\rFitting data in well [ {m+1}; {l+1}]", end="", flush=True)
        FRET_index = l + m * D_t_series_total_2
        fun = lambda x: FRET_decay_resid_2par(x, FRET[:, FRET_index], hours, index_start_fit2, index_end_fit2, k_inact_mean, k_0_fit1sel, F_min_fit1sel)
        result = least_squares(fun, x0)
        k_E3[m, l] = result.x[0]
        F_max[m, l] = result.x[1]

k_inact[:,:] = k_inact_mean
k_0[:,:] = k_0_fit1sel
F_min[:,:] = F_min_fit1sel

print(f"\n")

# Save results
savemat(
    mat_fit2_results_2par,
    {'k_E3': k_E3, 'F_max': F_max, 'k_inact': k_inact, 'k_0': k_0, 'F_min': F_min}
)


# # # =======================Option 2: 2plus1-parameter Fit =======================
# # # in this option, k_inact values for samples in file_name_2 are calculated using linear dependence:  k_inact = k_E3 * slope + intercept
# # # Values of the slope and intercept parameters are taken from Fit 1 performed on selected wells from file_name_1


print(f"Performing 2plus1-par fit on file {file_name_2} using time points from {hours[index_start_fit2]} to {hours[index_end_fit2]} hours")
print("k_inact is calculated as k_inact = k_E3 * slope + intercept")
print("The following parameters are fixed:")
print(f"k_0 = {np.round(k_0_fit1sel, 3)}; F_min = {np.round(F_min_fit1sel, 3)}")

plot_title = "Fit 2: 2plus1-parameter fit"

k_E3 = np.zeros((int(FRET.shape[1] / D_t_series_total_2), D_t_series_total_2))
F_max = np.zeros_like(k_E3)
k_inact = np.zeros_like(k_E3)
k_0 = np.zeros_like(k_E3)
F_min = np.zeros_like(k_E3)

x0 = np.ones(2)
for l in range(D_t_series_total_2):
    for m in range(int(FRET.shape[1] / D_t_series_total_2)):
        print(f"\rFitting data in well [ {m+1}; {l+1}]", end="", flush=True)
        FRET_index = l + m * D_t_series_total_2
        fun = lambda x: FRET_decay_resid_2plus1_par(
            x, FRET[:, FRET_index], hours, index_start_fit2, index_end_fit2,
            k_0_fit1sel, F_min_fit1sel, k_inact_vs_k_E3_slope, k_inact_vs_k_E3_intercept
        )
        result = least_squares(fun, x0)
        k_E3[m, l] = result.x[0]
        F_max[m, l] = result.x[1]
        k_inact[m, l] = k_E3[m, l] * k_inact_vs_k_E3_slope + k_inact_vs_k_E3_intercept

k_0[:,:] = k_0_fit1sel
F_min[:,:] = F_min_fit1sel

print(f"\n")

savemat(
    mat_fit2_results_2plus1par,
    {'k_E3': k_E3, 'F_max': F_max, 'k_inact': k_inact, 'k_0': k_0, 'F_min': F_min}
)


# # # ======================= Option 3: 3-parameter Fit =======================
# # # In this option, k_inact parameters are fitted individually for every sample in file_name_2


print(f"Performing 3-par fit on file {file_name_2} using time points from {hours[index_start_fit2]} to {hours[index_end_fit2]} hours")
print("The following parameters are fixed:")
print(f"k_0 = {np.round(k_0_fit1sel, 3)}; F_min = {np.round(F_min_fit1sel, 3)}")

plot_title = "Fit 2: 3-parameter fit"

k_E3 = np.zeros((int(FRET.shape[1] / D_t_series_total_2), D_t_series_total_2))
F_max = np.zeros_like(k_E3)
k_inact = np.zeros_like(k_E3)
k_0 = np.zeros_like(k_E3)
F_min = np.zeros_like(k_E3)

x0 = [1, 1, k_inact_mean]
for l in range(D_t_series_total_2):
    for m in range(int(FRET.shape[1] / D_t_series_total_2)):
        print(f"\rFitting data in well [ {m+1}; {l+1}]", end="", flush=True)
        FRET_index = l + m * D_t_series_total_2
        fun = lambda x: FRET_decay_resid_3par(x, FRET[:, FRET_index], hours, index_start_fit2, index_end_fit2, k_0_fit1sel, F_min_fit1sel)
        result = least_squares(fun, x0)
        k_E3[m, l] = result.x[0]
        F_max[m, l] = result.x[1]
        k_inact[m, l] = result.x[2]

k_0[:,:] = k_0_fit1sel
F_min[:,:] = F_min_fit1sel

print(f"\n")

savemat(
    mat_fit2_results_3par,
    {'k_E3': k_E3, 'F_max': F_max, 'k_inact': k_inact, 'k_0': k_0, 'F_min': F_min}
)


### =============================================================================
### SECTION 8: Generate a plot showing results of Fit 2 for selected wells
### =============================================================================

plt.figure(3, figsize=(6, 6))

exp_selection = []
sample_index = 0
color_index = 0

for i in col_selection_fit_2_plot:
    # Reset color index for each column (optional)
    color_index = 0
    # print(f"k_E3[:,{i}] = {np.round(k_E3[:, i], 3)}") # comment out if not needed
    for j in row_selection_fit_2_plot:
        sample_index = i + j * D_t_series_total_1
        exp_selection.append(sample_index)
        color = colors_rgb[color_index % len(colors_rgb)]
        color_index += 1

        # Plot experimental FRET values as scatter
        plt.scatter(
            hours[:],
            FRET[:, sample_index],
            s=5,
            color=color,
            facecolors='none',
            linewidths=0.5,
            label=f'Column {i+1}',
            clip_on=False
        )

        # Plot fitted decay curve
        plt.plot(
            hours[index_start_fit2:index_end_fit2],
            FRET_decay(
                hours[index_start_fit2:index_end_fit2],
                k_E3[j, i],
                F_max[j, i],
                k_inact[j, i],
                k_0[j, i],
                F_min[j, i]
            ),
            '-',
            color=color,
            linewidth=2,
            clip_on=False
        )

plt.xlabel('time (hours)')
plt.ylabel('FRET (au)')
plt.title(plot_title)

# Save the plot as a vector graphic
plt.savefig(fig_3_name, format="svg")

# Or display on screen (execution pauses until window is closed)
plt.show()
