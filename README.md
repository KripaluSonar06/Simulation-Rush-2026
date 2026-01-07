Solar Flare Pulse: Stochastic Signal Recovery via Bayesian MCMC
Project Overview
Solar Flare Pulse is a comprehensive Bayesian parameter inference solution for recovering physical parameters from noisy solar flare intensity observations. The project implements an adaptive Metropolis-Hastings Markov Chain Monte Carlo (MCMC) algorithm to estimate three key parameters: amplitude (A), quench time (τ), and frequency (ω).

Key Features
✅ State-of-the-art MCMC Algorithm: Adaptive Metropolis-Hastings with optimal proposal scaling

✅ Multi-chain Convergence Analysis: Gelman-Rubin diagnostic (R̂ < 1.05)

✅ Rigorous Uncertainty Quantification: Posterior distributions with credible intervals

✅ Publication-Quality Visualizations: 6 high-resolution diagnostic plots

✅ Complete Documentation: Technical report, guides, and inline code comments

✅ Production-Ready Code: Error handling, numerical safeguards, robust implementation

Table of Contents
Quick Start

System Requirements

Installation & Environment Setup

Project Structure

Running the Application

Output Files

Configuration

Troubleshooting

Citation & References

Quick Start
For the impatient, here's the fastest way to get results:

bash
# 1. Clone or download project
cd Simulation-Rush-2026

# 2. Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# 3. Install dependencies
pip install ..

# 4. Run MCMC analysis
python solar_flare_fixed.py

# 5. Generate visualizations
python generate_visualizations.py

# 6. View results
# Check output/ folder for plots and summary
Total time: ~7 minutes (5 min MCMC + 1 min visualization) (at max)

System Requirements
Hardware
CPU: Any modern processor (Intel, AMD, Apple Silicon)

RAM: Minimum 2 GB (4 GB recommended)

Storage: 500 MB available (for installation + outputs)

GPU: Not required (CPU-based computation)

Software
Operating System: Windows 10+, macOS 10.14+, Linux (any modern distribution)

Python: 3.9, 3.10, 3.11, 3.12, or 3.13

pip: Package installer for Python

Verify Installation
bash
python --version  # Should be 3.9+
pip --version     # Should be 21.0+
Installation & Environment Setup
Step 1: Install Python
Download and install Python from python.org:

Windows: Download .exe, run installer, check "Add Python to PATH"

macOS: Use Homebrew: brew install python3

Linux: Use package manager: apt-get install python3 python3-pip

Verify:

bash
python --version
Step 2: Clone/Download Project
Option A: Clone from GitHub (if using version control)

bash
git clone https://github.com/KripaluSonar06/Simulation-Rush-2026.git
cd Simulation-Rush-2026
Option B: Direct download

bash
# Download ZIP file, extract to desired location
cd Simulation-Rush-2026
Step 3: Create Virtual Environment
A virtual environment isolates dependencies for this project.

Windows:

bash
python -m venv env
env\Scripts\activate
macOS/Linux:

bash
python3 -m venv env
source env/bin/activate
Verify activation:

Command prompt should show (env) prefix

which python should show path inside env folder

Step 4: Install Dependencies
Option A: Using requirements.txt (Recommended)

bash
pip install -r requirements.txt
Option B: Manual installation

bash
pip install numpy==1.24.0
pip install scipy==1.10.0
pip install pandas==1.5.3
pip install matplotlib==3.5.3
pip install seaborn==0.12.2
pip install scikit-learn==1.2.0
pip install tqdm==4.65.0
Step 5: Verify Installation
bash
# Test imports
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"
python -c "import pandas; print(f'Pandas {pandas.__version__}')"
python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"

# All should print version numbers without errors
Project Structure
text
Solar-Flare-Pulse/
│
├── README.md                              ← This file
│
├── solar_flare.py                         ← MCMC analysis (RUN THIS FIRST)
├── generate_visualizations.py             ← Visualization generation (RUN SECOND)
│
├── flare_data.csv                         ← Input data (1000 observations)
│
├── Solar_Flare_Report.tex                 ← LaTeX report source
├── Solar_Flare_Report.pdf                 ← Compiled report (generate via Overleaf)
│
├── output/                                ← Results folder (created after running)
│   ├── trace_plots.png                    ← MCMC trace plots
│   ├── posterior_distributions.png        ← Posterior histograms
│   ├── corner_plot.png                    ← 2D posterior correlations
│   ├── convergence_diagnostics.png        ← Convergence analysis
│   ├── model_fit.png                      ← Data vs model fit
│   └── posterior_summary.txt               ← Statistical summary
│
├── mcmc_samples.npy                       ← Raw MCMC samples (created after step 1)
│

└── [other documentation files]
Running the Application
Execution Overview
The project runs in two stages:

text
Stage 1: MCMC Analysis
    ↓
    Generates mcmc_samples.npy
    ↓
    Prints convergence diagnostics
    ↓
    Reports posterior statistics

Stage 2: Visualization
    ↓
    Loads mcmc_samples.npy
    ↓
    Generates 5 plots
    ↓
    Creates text summary
Stage 1: Run MCMC Analysis
bash
python solar_flare_fixed.py
Expected Output:

text
================================================================================
SOLAR FLARE PULSE: STOCHASTIC SIGNAL RECOVERY
================================================================================

✓ Data loaded: 1000 observations
  Time range: [0.0000, 7.0000]
  Signal range: [-259.68, 228.01]

Initial parameters: A=0.8234, τ=5.0431, ω=9.8765

Chain 1/3
  Burn-in phase: 3000 iterations
  Main phase: 5000 iterations
  Acceptance rate: 0.234

Chain 2/3 ... [similar]
Chain 3/3 ... [similar]

================================================================================
CONVERGENCE DIAGNOSTICS
================================================================================

Gelman-Rubin Diagnostic R̂ (< 1.05 is ideal):
  A (Amplitude)     : 1.0023 ✓ PASS
  τ (Quench Time)   : 1.0012 ✓ PASS
  ω (Frequency)     : 1.0008 ✓ PASS

Effective Sample Size (ESS):
  A (Amplitude)     : 1342 (26.8% of total)
  τ (Quench Time)   : 1156 (23.1% of total)
  ω (Frequency)     : 1089 (21.8% of total)

================================================================================
POSTERIOR STATISTICS
================================================================================

A (Amplitude):
  MAP estimate:           0.825123
  Posterior mean:         0.825123
  Posterior std dev:      0.053421
  95% CI:                 [0.721, 0.931]

[Similar for τ and ω...]

✓ MCMC Analysis Complete!
✓ Samples saved to mcmc_samples.npy
Duration: ~4-5 minutes (depending on system)

Outputs Created:

mcmc_samples.npy (5000 samples × 3 parameters)

Console output showing convergence diagnostics

Stage 2: Generate Visualizations
bash
python generate_visualizations_FINAL.py
Expected Output:

text
================================================================================
SOLAR FLARE MCMC VISUALIZATION & RESULTS GENERATION
================================================================================

✓ Output directory created: ./output

================================================================================
LOADING MCMC RESULTS
================================================================================

✓ MCMC samples loaded: (5000, 3)
  - Total samples: 5000
  - Parameters: 3 (A, tau, omega)
✓ Flare data loaded: 1000 observations

================================================================================
GENERATING VISUALIZATIONS
================================================================================

Generating Trace Plots...
✓ Trace plots saved: ./output/trace_plots.png

Generating Posterior Distributions...
✓ Posterior distributions saved: ./output/posterior_distributions.png

Generating Corner Plot...
✓ Corner plot saved: ./output/corner_plot.png

Generating Convergence Diagnostics...
✓ Convergence diagnostics saved: ./output/convergence_diagnostics.png

Generating Model Fit Plot...
✓ Model fit plot saved: ./output/model_fit.png

Generating Summary Report...
✓ Summary report saved: ./output/posterior_summary.txt

================================================================================
✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!
================================================================================

Output directory: /full/path/to/output

Generated files:
  ✓ corner_plot.png
  ✓ convergence_diagnostics.png
  ✓ model_fit.png
  ✓ posterior_distributions.png
  ✓ posterior_summary.txt
  ✓ trace_plots.png
Duration: ~30-60 seconds

Outputs Created:

output/trace_plots.png

output/posterior_distributions.png

output/corner_plot.png

output/convergence_diagnostics.png

output/model_fit.png

output/posterior_summary.txt

Full Workflow Command
Run both stages sequentially:

bash
python solar_flare_fixed.py && python generate_visualizations_FINAL.py
Output Files
1. Trace Plots (trace_plots.png)
Shows: Parameter evolution across MCMC iterations

Interpretation:

Blue line: Raw MCMC trace

Red dashed line: Posterior mean

Green shading: 95% credible interval

Look for: Stable, horizontal lines (no trends)

2. Posterior Distributions (posterior_distributions.png)
Shows: Probability distribution of each parameter

Interpretation:

Blue histogram: Posterior density

Red line: Kernel density estimate

Red dashed line: Posterior mean

Green line: Posterior median

Yellow shading: 95% credible interval

Look for: Unimodal, symmetric distributions

3. Corner Plot (corner_plot.png)
Shows: 1D and 2D marginal posteriors

Interpretation:

Diagonal: 1D marginal posteriors

Off-diagonal: 2D scatter plots (joint posteriors)

Red crosses: Posterior means

Look for: Minimal correlation between parameters

4. Convergence Diagnostics (convergence_diagnostics.png)
Shows: Three diagnostic panels per parameter

Interpretation:

Left: Running mean (should stabilize)

Middle: Autocorrelation function (should decay rapidly)

Right: Cumulative mean (should converge to final value)

Look for: Running mean stable after 500 iterations, ACF decay within 30 lags

5. Model Fit (model_fit.png)
Shows: Observed data vs model predictions

Interpretation:

Gray dots: Observed flare intensity

Red line: Maximum a posteriori (MAP) fit

Red shading: 90% posterior prediction interval

Top: Full time range

Bottom: Zoomed view around peak

6. Summary Report (posterior_summary.txt)
Shows: Comprehensive statistics text file

Contents:

Point estimates (mean, median, mode)

Uncertainty measures (std, MAD, IQR)

Credible intervals (68%, 95%)

Distribution summaries (percentiles)

Shape descriptors (skewness, kurtosis)

Configuration
Modifying MCMC Parameters
Edit solar_flare_fixed.py:

python
# Around line 50-80, find:

N_CHAINS = 3                    # Number of independent chains
N_ITERATIONS = 8000             # Iterations per chain
N_BURN_IN = 3000                # Burn-in iterations
N_MAIN = 5000                   # Main phase iterations
INITIAL_SD = 0.1                # Initial proposal standard deviation
ADAPT_EVERY = 100               # Adapt covariance every N iterations

# Modify as needed:
N_CHAINS = 5                    # Run 5 chains instead of 3
N_ITERATIONS = 10000            # More iterations for precision
Modifying Data Input
Default: Uses flare_data.csv

To use different data:

python
# Line ~100, find:
data = pd.read_csv('flare_data.csv')

# Change to:
data = pd.read_csv('your_data.csv')
Data Format Required:

text
t,s
0.0000,5.234
0.0050,6.123
...
7.0000,-2.345
Columns:

t: Time (float, arbitrary units)

s: Flare intensity (float, arbitrary units)

Modifying Output Directory
Edit generate_visualizations_FINAL.py:

python
# Line ~25, find:
output_dir = Path('./output')

# Change to:
output_dir = Path('./results')  # Different folder name
Adjusting Plot Resolution
Edit generate_visualizations_FINAL.py:

python
# In plot functions, find:
plt.savefig(file, dpi=300, bbox_inches='tight')

# Change dpi:
plt.savefig(file, dpi=150, bbox_inches='tight')  # Lower quality, smaller file
plt.savefig(file, dpi=600, bbox_inches='tight')  # Higher quality, larger file
Troubleshooting
Issue: ModuleNotFoundError: No module named 'numpy'
Solution:

bash
# Verify virtual environment is activated (should show (env) prefix)
# If not:
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows

# Then install:
pip install -r requirements.txt
Issue: FileNotFoundError: flare_data.csv
Solution:

bash
# Ensure you're in the correct directory:
pwd                      # macOS/Linux
cd %cd%                 # Windows

# Should be in Solar-Flare-Pulse root directory
# Verify flare_data.csv exists:
ls flare_data.csv       # macOS/Linux
dir flare_data.csv      # Windows
Issue: FileNotFoundError: mcmc_samples.npy
Solution:

bash
# Run Stage 1 first (MCMC analysis):
python solar_flare_fixed.py

# This creates mcmc_samples.npy
# Then run Stage 2:
python generate_visualizations_FINAL.py
Issue: UnicodeEncodeError on Windows
Solution:
This is already fixed in generate_visualizations_FINAL.py. If using original script, it uses UTF-8 encoding which works on all systems.

Issue: Script runs but produces no output
Solution:

bash
# Verify file was modified
cat solar_flare_fixed.py | head -50

# Check Python is working:
python -c "print('Python works')"

# Run with explicit error output:
python -u solar_flare_fixed.py 2>&1 | head -50
Issue: Very slow execution
Common causes:

Running other applications (close them)

Using spinning disk (SSD is faster)

Low RAM (close browser, etc.)

Solution:

Wait 5-6 minutes (normal for MCMC)

Run on faster system if available

Reduce N_ITERATIONS in configuration

Issue: Different results on multiple runs
Expected behavior: MCMC is stochastic, small variations are normal

To get reproducible results:

python
# Add at top of solar_flare_fixed.py:
import numpy as np
np.random.seed(42)
Dependencies
All dependencies listed in requirements.txt:

text
numpy==1.24.0              # Numerical computing
scipy==1.10.0              # Scientific computing (optimization, statistics)
pandas==1.5.3              # Data manipulation
matplotlib==3.5.3          # Plotting library
seaborn==0.12.2            # Statistical data visualization
scikit-learn==1.2.0        # Machine learning utilities
tqdm==4.65.0               # Progress bars
Why Each Library?
Library	Purpose
NumPy	Multi-dimensional arrays, matrix operations
SciPy	Optimization (L-BFGS-B), statistics, special functions
Pandas	CSV file I/O, DataFrame operations
Matplotlib	Static plotting (trace plots, posteriors, etc.)
Seaborn	Enhanced matplotlib styling and color palettes
scikit-learn	Median absolute deviation computation
tqdm	Progress bars for long computations
Citation & References
Cite This Work
text
@software{solarflare2026,
  title={Solar Flare Pulse: Stochastic Signal Recovery via Bayesian MCMC},
  author={Bayesian Inference Solutions},
  year={2026},
  organization={Simulation Rush 2026},
  url={https://github.com/yourusername/solar-flare-pulse}
}
Key References
Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC. Journal of Computational and Graphical Statistics, 18(2), 349-367.

Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.). Chapman and Hall/CRC.

Hudson, H. S. (2011). Global properties of solar flares. Living Review in Solar Physics, 8(1), 2-59.

Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.

Online Resources
Python Documentation: https://docs.python.org/3/

NumPy Documentation: https://numpy.org/doc/

SciPy Documentation: https://docs.scipy.org/

Matplotlib Documentation: https://matplotlib.org/

MCMC Theory: https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo

Performance Metrics
Computational Cost
Stage	Duration	Hardware
Data loading	< 1 second	Any
Parameter optimization	~5 seconds	CPU
MCMC sampling (3 chains)	~4-5 minutes	CPU
Visualization generation	~30-60 seconds	CPU/GPU
Total	~6 minutes	Any modern computer
Memory Usage
Peak RAM: ~200 MB

Output files: ~1.3 MB total (6 plots + 1 summary)

Raw samples: ~120 KB

Getting Help
Common Questions
Q: How long does MCMC take?
A: ~5 minutes for 24,000 total samples (3 chains × 8,000 iterations). This is normal.

Q: Can I run on GPU?
A: Current implementation is CPU-only. GPU acceleration possible with CuPy/JAX but not implemented.

Q: What if I get different results?
A: MCMC is stochastic. Results vary slightly but should be similar. Set random seed for reproducibility.

Q: Can I use different data?
A: Yes, modify flare_data.csv or point to different file. Ensure same CSV format.

Q: How do I interpret the plots?
A: See Output Files section above.

Support
For issues:

Check Troubleshooting section

Read inline code comments in Python scripts

Review docs/ folder for technical guides

Check OFFICIAL_CONFIRMATION.md for verification

Advanced Usage
Running with Custom Priors
Edit solar_flare_fixed.py, find prior definitions and modify:

python
# Current priors (uniform):
prior_A = lambda x: 1.0 if 0 < x < 2 else -np.inf
prior_tau = lambda x: 1.0 if 1 < x < 10 else -np.inf
prior_omega = lambda x: 1.0 if 1 < x < 20 else -np.inf

# Change to Gaussian priors (example):
prior_A = lambda x: stats.norm.logpdf(x, loc=1.0, scale=0.5)
prior_tau = lambda x: stats.norm.logpdf(x, loc=5.0, scale=2.0)
prior_omega = lambda x: stats.norm.logpdf(x, loc=10.0, scale=3.0)
Batch Processing Multiple Datasets
bash
# Create loop in shell script:
for dataset in data1.csv data2.csv data3.csv; do
    # Modify line 100 of solar_flare_fixed.py:
    sed -i "s/flare_data.csv/$dataset/g" solar_flare_fixed.py
    
    # Run analysis:
    python solar_flare.py
    
    # Save results:
    mv mcmc_samples.npy "samples_${dataset%.csv}.npy"
    mv output "results_${dataset%.csv}"
done
Deactivating Virtual Environment
When finished, deactivate the virtual environment:

bash
deactivate
Command prompt will return to normal (no (env) prefix).

License & Attribution
This project is provided for educational and competitive purposes. Please cite appropriately when using in publications or presentations.


Ready to win Simulation Rush 2026! 🏆
