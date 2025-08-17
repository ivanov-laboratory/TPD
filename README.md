# Single-Assay Characterization of Ternary Complex Assembly and Activity in Targeted Protein Degradation

**Copyright (c) 2025 Dmitri Ivanov**

---

## Overview
Targeted protein degradation (TPD) is a rapidly advancing therapeutic strategy that selectively eliminates disease-associated proteins by co-opting the cell’s protein degradation machinery. Covalent modification of proteins with ubiquitin is a critical event in TPD, yet analytical tools for quantifying ubiquitination kinetics have been limited.

Here, we present a real-time, high-throughput fluorescent assay utilizing purified, FRET-active E2-Ub conjugates to monitor ubiquitin transfer. This assay is highly versatile, requiring no engineering of the target protein or ligase, thereby accelerating assay development and minimizing artifacts. The single-step, single-turnover nature of the monitored reaction enables rigorous and quantitative analysis of ubiquitination kinetics.

We demonstrate that this assay can measure key degrader characteristics, including:
- Degrader affinity for the target protein
- Degrader affinity for the ligase
- Affinity of ternary complex assembly
- Catalytic efficiency of the ternary complex

This comprehensive, single-assay approach provides high sensitivity and accuracy for ternary complex characterization, empowering discovery and optimization of heterobifunctional degraders and molecular glues.

---

## Repository Contents
This repository contains sample datasets and Python scripts for quantitative analysis of ubiquitination kinetics using FRET-active E2-Ub conjugates.

Data analysis proceeds in two main steps:

1. **Extract catalytic rates from FRET decay datasets**  
   (`get_rates.py`)

2. **Analyze catalytic rates as a function of reagent concentrations**  
   (`glue_fit_rates.py` or `protac_fit_rates.py`)

Each step uses a **solver-in-the-loop strategy**, in which a numerical solver generates model predictions for trial parameter sets. Predictions are compared to experimental data using nonlinear least-squares optimization (`scipy.optimize.least_squares`), and parameters are iteratively refined to minimize the residuals.

---

## Step 1: FRET Decay Fitting (`get_rates.py`)
- **Input**: Donor and acceptor intensities parsed from plate-reader text files.  
- **Processing**: FRET is computed as acceptor/donor.  
- **Model**: For each sample, FRET decay is modeled by a time-dependent inactivation (TDI) ODE, numerically integrated using `scipy.integrate.solve_ivp`.  
- **Fitting strategy**:  
  - **Fit 1 (5-parameter fit)**:  
    - Parameters: `k_E3`, `F_max`, and `k_inact` fitted per sample; `k_0` and `F_min` fitted globally.  
    - Recommended for long acquisitions (4–6 h) on a subset of wells or applied column-by-column for the entire plate.  
  - **Fit 2 (reduced fit)**:  
    - For shorter traces (1–2 h) or large datasets.  
    - Globally determined parameters from Fit 1 are fixed; remaining 2 or 3 parameters refitted per sample.  
    - Options: 2-parameter, 2-plus-1 (with `k_inact` constrained linearly vs. `k_E3`), or 3-parameter variants.  

---

## Step 2: Equilibrium Model Fitting (`protac_fit_rates.py` and `glue_fit_rates.py`)
- **Input**: Catalytic rates obtained from Step 1.  
- **Model**: Mass-action equilibrium models of ternary complex formation.  
  - `protac_fit_rates.py`: for heterobifunctional degraders (PROTACs).  
  - `glue_fit_rates.py`: for molecular glues.  
- **Computation**:  
  - Algebraic systems enforcing mass conservation and binding equilibria are solved numerically with `scipy.optimize.fsolve`.  
  - Solutions yield concentrations of the catalytically active species.  
  - These numeric predictions populate a model rate matrix compared against experimental catalytic rate matrices in `least_squares`.

---

## Dependencies
- Python ≥ 3.9  
- [NumPy](https://numpy.org/)  
- [SciPy](https://scipy.org/)  
- [Matplotlib](https://matplotlib.org/)  

---

## Citation
If you use this repository in your work, please cite:

> Ivanov, D. *Single-Assay Characterization of Ternary Complex Assembly and Activity in Targeted Protein Degradation*. (2025).

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
