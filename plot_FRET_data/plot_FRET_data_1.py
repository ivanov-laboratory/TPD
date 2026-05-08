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
This script reads raw fluorescence time-course data exported from a BioTek Synergy Neo2 plate reader
and generates a plot of selected wells for quick visual inspection.

Workflow:
- Loads donor (fluorescein) and acceptor (AF594) fluorescence intensities from the file.
- Converts time values to hours and calculates FRET as acceptor/donor intensity ratios.
- Allows selection of specific rows and columns (wells) for plotting.
- Plots FRET vs. time for the chosen wells using predefined colors.
- Saves the plot as an SVG file and optionally displays it on screen.
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

# Name of the text file containing the raw FRET data exported from the plate reader.
# This file should contain donor and acceptor fluorescence readings in tab-delimited format,
# as exported by the BioTek Synergy Neo2 or a similar plate reader.
file_name = "Poma_SALL4_aa378_453_exp03.txt"

# use the same version number as in the title of this script to be added to all output files of this script
script_version_number = "1"

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
#   - Each line corresponds to a single timepoint in the kinetic run.
#   - The readings for all measured wells are written into a single line row-by-row, left to right.
#     Example: A1, A2, A3, ..., B1, B2, B3, ...
#
# IMPORTANT: Python uses zero-based indexing!
# For example, well A1 is row_index=0, col_index=0.
# Well B1 is row_index=1, col_index=0, and so on.
# ----------------------------------------------------------------------

# Number of degrader titration series (columns) measured in the exported data.
# This corresponds to the number of columns in the rectangular area read by the plate reader.
# The value can vary between experiments.
D_t_series_total = 7

# ----------------------------------------------------------------------
# Selection for plotting and fitting:
#
# These parameters control which wells will be plotted in the quick visualization section
# and optionally used for initial kinetic parameter fitting.
#
# col_selection_plot : List of column indices (0-based) to include.
# row_selection_plot : List of row indices (0-based) to include.
#
# Example:
#   col_selection_plot = [3]   → only use column #4 (since Python counts from 0).
#   row_selection_plot = [0, 1, 2] → use the first three rows (A, B, C).
#
# To include all 12 rows (A–L), you can use:
#   row_selection_plot = list(range(12))
# ----------------------------------------------------------------------
col_selection_plot = [0,1,2,3,4,5,6]
row_selection_plot = [2]

### =============================================================================
### SECTION 2: Function Definitions
### =============================================================================

def split_text_into_sections(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove lines that start with '#' or are empty
    filtered_lines = [line for line in lines if not line.strip().startswith('#')]

    # Join the remaining lines into a single string
    text = ''.join(filtered_lines)

    # Split the text into sections based on two or more newlines
    sections = text.strip().split('\n\n')
    
    return sections

def time_to_hours(time_str):
    # # Convert time in HH:MM:SS format to hours as a float.
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600

def load_data_from_string(data_string):
    # # Load data from a string, converting time and skipping the temperature entry in the second column.
    
    # Create a file-like object from the string
    data_file = StringIO(data_string)
    
    # Determine the number of columns by reading the first line
    first_line = data_file.readline()
    num_columns = len(first_line.strip().split('\t'))
    
    # Reset the file pointer to the beginning of the file-like object
    data_file.seek(0)
    
    # Generate column indices to read, skipping the second column (index 1)
    usecols = tuple(i for i in range(num_columns) if i != 1)

      # Prepare a list to hold valid lines
    valid_lines = []
    
    for line in data_file:
        # Split the line into columns
        columns = line.strip().split('\t')
        
        # Check if the number of columns matches the expected count
        if len(columns) != num_columns:
            print(f"Stopping read: Line does not have the expected number of entries:\n{line}")
            break
        
        # Append valid lines to the list
        valid_lines.append(line)
    
    # Convert the valid lines list back to a StringIO object for numpy to read
    valid_data_string = StringIO("\n".join(valid_lines))
    
    # Define converters to apply the time conversion function to the first column (index 0)
    converters = {0: time_to_hours}
    
    # Load the data using numpy.loadtxt
    data = np.loadtxt(
        valid_data_string, 
        delimiter='\t',  # use tab as delimiter
        usecols=usecols, 
        converters=converters
    )
    
    return data

colors_rgb = [
    (0.6, 0.6, 0.6),    
    (0.96, 0.043, 0.043),
    (0.98, 0.62, 0.055),
    (0.98039216, 0.88235294, 0.05490196),
    (0.77254902, 0.69803922, 0.07058824),
    (0.54901961, 0.83921569, 0.09411765),
    (0.11372549, 1.        , 0.55686275),
    (0.0627451 , 0.62352941, 0.34509804),
    (0.04705882, 0.6       , 0.69803922),
    (0.0745098 , 0.85882353, 1.        ),
    (0.16078431, 0.10588235, 1.        ),
    (0.40784314, 0.08627451, 0.61960784),
    (0.0, 0.0, 0.0)
]


### =============================================================================
### SECTION 3: Open file and calculate FRET values
### =============================================================================

# Open file and split into sections using double newline as delimiter
sections = split_text_into_sections(file_name)

# Load donor (fluoresceine) fluorescence intensities from sections[1]
data_528 = load_data_from_string(sections[1])

# Load acceptor (AF594) fluorescence intensities from sections[3]
data_635 = load_data_from_string(sections[3])

# read time values (hours) from the first column
hours=data_528[:,0]

# create FRET array by dividing the acceptor array by donor array and omitting first column containing time stamps
FRET = np.divide(data_635[:,1:], data_528[:,1:])

### =============================================================================
### SECTION 4: Plot selected wells from file_1
### =============================================================================
### The purpose of this section is just to visualise a selection of the data file
### to quickly check whether the data look OK

plt.figure(1,figsize=(6, 6))

exp_selection = []
sample_index = 0

color_index = 0
for i in col_selection_plot:
    # # uncomment line below if you want colors to restart for every column plotted
    # color_index = 0
    for j in row_selection_plot:
        sample_index=(i + j * D_t_series_total)  # Append calculated value to the list
        color=colors_rgb[color_index % len(colors_rgb)]
        exp_selection.append(i + j * D_t_series_total)
        line, = plt.plot(hours[:], FRET[:, sample_index],color=color,label=f'Row {j+1} Col {i+1}')
        line.set_clip_on(False)
        # plt.scatter(hours[:], FRET[:, sample_index], s=10, color=color, facecolors='white', linewidths=0.5, label=f'Row {j+1} Col {i+1}', clip_on=False)
        color_index+=1
    


plt.xlabel('hours')
plt.ylabel('FRET')
# # plt.title('FRET vs Hours for Selected Columns')
plt.legend()
# plt.grid(True)
plt.gca().set_xlim([0, 4])
plt.gca().set_ylim([0.0, 0.7])

# # the plot can either be saved into a vector graphic file to make into a figure
# # edit command below to generate a custom output filename
plt.savefig(os.path.splitext(file_name)[0]+"_plot_selection_" + script_version_number + ".svg", format="svg")
plt.savefig(os.path.splitext(file_name)[0]+"_plot_selection_" + script_version_number + ".eps", format="eps")

# # or it can be output to the screen. the execution of the script pauses until th figure window is closed
plt.show()
