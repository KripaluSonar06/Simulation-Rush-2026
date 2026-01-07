# 🌞 Solar Flare Pulse

### Bayesian Parameter Inference via Adaptive MCMC

This repository contains a **Bayesian parameter inference pipeline** for recovering physical parameters of a solar flare pulse model from noisy observational data. The project applies an **Adaptive Metropolis–Hastings Markov Chain Monte Carlo (MCMC)** algorithm to estimate model parameters with rigorous uncertainty quantification and convergence diagnostics.

Developed as part of **Simulation Rush 2026**.

---

## 📌 Project Overview

Solar flare intensity signals are noisy and highly non-linear, making deterministic estimation unreliable. This project formulates the problem probabilistically and estimates the following physical parameters:

* **Amplitude** (A)
* **Quench Time** (\tau)
* **Oscillation Frequency** (\omega)

The framework produces statistically valid posterior distributions along with diagnostic visualizations.

---

## ✨ Key Features

* Adaptive Metropolis–Hastings MCMC with optimal proposal scaling
* Multi-chain convergence analysis (Gelman–Rubin statistic)
* Posterior uncertainty quantification with credible intervals
* Publication-quality diagnostic plots
* Clean, reproducible, and well-documented Python implementation

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/KripaluSonar06/Simulation-Rush-2026.git
cd Simulation-Rush-2026

# Create and activate virtual environment
python -m venv env
source env/bin/activate     # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run MCMC analysis
python solar_flare_mcmc.py

# Generate visualizations
python generate_visualization.py
```

All results are saved in the `output/` directory.

---

## 🖥 System Requirements

**Hardware**

* CPU: Any modern processor
* RAM: ≥ 2 GB (4 GB recommended)
* GPU: Not required

**Software**

* Python 3.9 or higher
* pip 21+

---

## 📂 Project Structure

```text
Simulation-Rush-2026/
│
├── solar_flare_mcmc.py          # Main MCMC inference script
├── generate_visualization.py   # Visualization & diagnostics
├── flare_data.csv              # Input dataset
│
├── output/                     # Generated after execution
│   ├── trace_plots.png
│   ├── posterior_distributions.png
│   ├── corner_plot.png
│   ├── convergence_diagnostics.png
│   ├── model_fit.png
│   └── posterior_summary.txt
│
├── Solar_Flare_Report.tex       # Technical report (LaTeX)
├── requirements.txt
└── README.md
```

---

## ⚙️ Running the Pipeline

The workflow has **two stages**:

### 1️⃣ MCMC Inference

```bash
python solar_flare_mcmc.py
```

* Runs multi-chain adaptive MCMC
* Prints convergence diagnostics (R̂, ESS)
* Saves samples to `mcmc_samples.npy`

### 2️⃣ Visualization & Analysis

```bash
python generate_visualization.py
```

* Generates diagnostic plots
* Produces posterior summary statistics
* Stores results in `output/`

---

## 📊 Output Files

| File                          | Description                       |
| ----------------------------- | --------------------------------- |
| `trace_plots.png`             | MCMC trace and mixing behavior    |
| `posterior_distributions.png` | Marginal posterior distributions  |
| `corner_plot.png`             | Joint posterior correlations      |
| `convergence_diagnostics.png` | Autocorrelation and running means |
| `model_fit.png`               | Observed data vs MAP model        |
| `posterior_summary.txt`       | Numerical posterior statistics    |

---

## 🔧 Configuration

MCMC parameters can be tuned in `solar_flare_mcmc.py`:

```python
N_CHAINS = 3
N_ITERATIONS = 8000
N_BURN_IN = 3000
ADAPT_EVERY = 100
```

To use a different dataset, replace `flare_data.csv` with your own file:

```text
t,s
0.000,5.234
0.005,6.123
...
```

---

## 🧪 Reproducibility

MCMC is stochastic. For reproducible runs, set a random seed in `solar_flare_mcmc.py`:

```python
import numpy as np
np.random.seed(42)
```

---


## 📚 References

* Roberts & Rosenthal (2009) — *Examples of Adaptive MCMC*
* Gelman et al. (2013) — *Bayesian Data Analysis*
* Hastings (1970) — *Monte Carlo Sampling Methods*
* Hudson (2011) — *Global Properties of Solar Flares*

---

## 📜 License

This project is intended for **educational, academic, and competitive use**.
Please cite appropriately if used in research or publications.

---


