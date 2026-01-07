import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory created: {output_dir}")

# LOAD DATA & RESULTS
def load_results(samples_file='mcmc_samples.npy', data_file='flare_data.csv'):
    """Load MCMC samples and original data"""

    print("\n" + "="*80)
    print("LOADING MCMC RESULTS")
    print("="*80)

    # Load MCMC samples
    try:
        samples = np.load(samples_file)
        print(f"✓ MCMC samples loaded: {samples.shape}")
        print(f"  - Total samples: {samples.shape[0]}")
        print(f"  - Parameters: {samples.shape[1]} (A, tau, omega)")
    except FileNotFoundError:
        print(f"✗ Error: {samples_file} not found")
        print("  Run 'python solar_flare_fixed.py' first to generate MCMC results")
        return None, None

    # Load original data
    try:
        data = pd.read_csv(data_file)
        t_data = data['t'].values
        y_data = data['s'].values
        print(f"✓ Flare data loaded: {len(t_data)} observations")
    except FileNotFoundError:
        print(f"✗ Error: {data_file} not found")
        t_data, y_data = None, None

    return samples, (t_data, y_data)

# 1. TRACE PLOTS - Parameter Evolution
def plot_trace_plots(samples, output_dir):

    print("\n" + "-"*80)
    print("GENERATING TRACE PLOTS")
    print("-"*80)

    param_names = ['A (Amplitude)', 'tau (Quench Time)', 'omega (Frequency)']
    param_symbols = ['$A$', r'$\tau$', r'$\omega$']
    n_params = samples.shape[1]
    iterations = np.arange(samples.shape[0])

    # Create figure with 3 subplots (one per parameter)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('MCMC Trace Plots - Parameter Evolution Across Iterations',
                 fontsize=16, fontweight='bold', y=0.995)

    for i, (ax, name, symbol) in enumerate(zip(axes, param_names, param_symbols)):
        # Plot trace
        ax.plot(iterations, samples[:, i], linewidth=0.5, alpha=0.7, color='steelblue')

        # Add mean line
        mean_val = np.mean(samples[:, i])
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_val:.4f}', alpha=0.8)

        # Add credible interval shading
        ci_lower = np.percentile(samples[:, i], 2.5)
        ci_upper = np.percentile(samples[:, i], 97.5)
        ax.fill_between([0, len(samples)], ci_lower, ci_upper,
                        color='green', alpha=0.1, label='95% CI')

        # Labels and formatting
        ax.set_ylabel(f'{symbol}\n{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration' if i == 2 else '', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add statistics box
        stats_text = (f'Mean: {mean_val:.6f}\n'
                     f'Std: {np.std(samples[:, i]):.6f}\n'
                     f'Min: {np.min(samples[:, i]):.6f}\n'
                     f'Max: {np.max(samples[:, i]):.6f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[0].set_xlim(0, len(samples))

    plt.tight_layout()

    # Save figure
    trace_file = output_dir / 'trace_plots.png'
    plt.savefig(trace_file, dpi=300, bbox_inches='tight')
    print(f"✓ Trace plots saved: {trace_file}")
    plt.close()

# 2. POSTERIOR DISTRIBUTIONS - Histograms
def plot_posterior_distributions(samples, output_dir):
    print("\n" + "-"*80)
    print("GENERATING POSTERIOR DISTRIBUTIONS")
    print("-"*80)

    param_names = ['A (Amplitude)', 'tau (Quench Time)', 'omega (Frequency)']
    param_symbols = ['$A$', r'$\tau$', r'$\omega$']
    param_bounds = [(0, 2), (1, 10), (1, 20)]
    n_params = samples.shape[1]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Posterior Distributions with Credible Intervals',
                 fontsize=16, fontweight='bold')

    for i, (ax, name, symbol, bounds) in enumerate(zip(axes, param_names, param_symbols, param_bounds)):
        param_samples = samples[:, i]

        # Histogram
        counts, bins, patches = ax.hist(param_samples, bins=50, density=True,
                                        alpha=0.6, color='steelblue',
                                        edgecolor='black', linewidth=0.5)

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(param_samples)
            x_range = np.linspace(param_samples.min(), param_samples.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE', alpha=0.8)
        except:
            pass

        # Statistics
        mean_val = np.mean(param_samples)
        median_val = np.median(param_samples)
        std_val = np.std(param_samples)
        ci_lower = np.percentile(param_samples, 2.5)
        ci_upper = np.percentile(param_samples, 97.5)

        # Add vertical lines
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2.5,
                  label=f'Mean = {mean_val:.4f}', alpha=0.8)
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=2.5,
                  label=f'Median = {median_val:.4f}', alpha=0.8)

        # Shade 95% credible interval
        ax.axvspan(ci_lower, ci_upper, alpha=0.15, color='yellow',
                  label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')

        # Labels and formatting
        ax.set_xlabel(f'{symbol}\n{name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11)
        ax.set_xlim(bounds)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best', fontsize=9, framealpha=0.95)

        # Statistics text box
        stats_text = (f'Mean: {mean_val:.6f}\n'
                     f'Median: {median_val:.6f}\n'
                     f'Std: {std_val:.6f}\n'
                     f'Mode: {bins[np.argmax(counts)]:.6f}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    # Save figure
    posterior_file = output_dir / 'posterior_distributions.png'
    plt.savefig(posterior_file, dpi=300, bbox_inches='tight')
    print(f"✓ Posterior distributions saved: {posterior_file}")
    plt.close()

# 3. CORNER PLOT - 2D Posterior Correlations

def plot_corner_plot(samples, output_dir):

    print("\n" + "-"*80)
    print("GENERATING CORNER PLOT")
    print("-"*80)

    param_names = ['$A$ (Amplitude)', r'$\tau$ (Quench)', r'$\omega$ (Frequency)']
    n_params = samples.shape[1]

    # Use every 10th sample for speed (corner plot can be slow)
    samples_thin = samples[::10, :]

    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    fig.suptitle('Corner Plot - Parameter Correlations',
                 fontsize=14, fontweight='bold', y=0.995)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(samples[:, i], bins=30, density=True,
                       alpha=0.6, color='steelblue', edgecolor='black')
                ax.set_ylabel('Density', fontsize=9)

                # Mean line
                mean_val = np.mean(samples[:, i])
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5)

            else:
                # Off-diagonal: 2D scatter
                ax.scatter(samples_thin[:, j], samples_thin[:, i],
                          alpha=0.3, s=10, color='steelblue')

                # Add mean cross
                mean_i = np.mean(samples[:, i])
                mean_j = np.mean(samples[:, j])
                ax.axhline(mean_i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.axvline(mean_j, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=10, fontweight='bold')
            if j == 0 and i != 0:
                ax.set_ylabel(param_names[i], fontsize=10, fontweight='bold')

            # Fixed: set_ticklabels (plural)
            if i != 0 or j != 0:
                ax.get_xaxis().set_ticklabels([])
                ax.get_yaxis().set_ticklabels([])

            ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save figure
    corner_file = output_dir / 'corner_plot.png'
    plt.savefig(corner_file, dpi=300, bbox_inches='tight')
    print(f"✓ Corner plot saved: {corner_file}")
    plt.close()

# 4. CONVERGENCE DIAGNOSTICS

def plot_convergence_diagnostics(samples, output_dir):

    print("\n" + "-"*80)
    print("GENERATING CONVERGENCE DIAGNOSTICS")
    print("-"*80)

    param_names = ['A (Amplitude)', 'tau (Quench Time)', 'omega (Frequency)']
    param_symbols = ['$A$', r'$\tau$', r'$\omega$']
    n_params = samples.shape[1]

    fig = plt.figure(figsize=(15, 12))

    for i in range(n_params):
        param_samples = samples[:, i]

        # Row 1: Running mean
        ax1 = plt.subplot(n_params, 3, 3*i + 1)
        running_mean = np.cumsum(param_samples) / np.arange(1, len(param_samples) + 1)
        ax1.plot(running_mean, linewidth=1, color='steelblue', alpha=0.7)
        ax1.set_ylabel(param_symbols[i], fontsize=11, fontweight='bold')
        ax1.set_title(f'{param_names[i]} - Running Mean', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(samples))

        # Row 2: Autocorrelation
        ax2 = plt.subplot(n_params, 3, 3*i + 2)
        max_lag = min(200, len(samples) // 4)
        acf_vals = np.zeros(max_lag)
        centered = param_samples - np.mean(param_samples)
        c0 = np.dot(centered, centered) / len(centered)

        for lag in range(max_lag):
            if lag == 0:
                acf_vals[lag] = 1.0
            else:
                c_lag = np.dot(centered[:-lag], centered[lag:]) / len(centered)
                acf_vals[lag] = c_lag / c0

        ax2.bar(range(max_lag), acf_vals, color='steelblue', alpha=0.7, width=1)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% CI')
        ax2.axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_title(f'{param_names[i]} - Autocorrelation', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Lag', fontsize=9)
        ax2.set_ylabel('ACF', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=8)

        # Row 3: Cumulative mean
        ax3 = plt.subplot(n_params, 3, 3*i + 3)
        cumsum = np.cumsum(param_samples)
        cumulative_mean = cumsum / np.arange(1, len(param_samples) + 1)
        ax3.plot(cumulative_mean, linewidth=1.5, color='steelblue', alpha=0.7, label='Cumulative Mean')
        ax3.axhline(np.mean(param_samples), color='red', linestyle='--', linewidth=2,
                   label=f'Final Mean: {np.mean(param_samples):.4f}')
        ax3.set_title(f'{param_names[i]} - Convergence', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Iteration', fontsize=9)
        ax3.set_ylabel(param_symbols[i], fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, len(samples))

    plt.suptitle('Convergence Diagnostics - Running Means, Autocorrelation, Convergence',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    diagnostic_file = output_dir / 'convergence_diagnostics.png'
    plt.savefig(diagnostic_file, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence diagnostics saved: {diagnostic_file}")
    plt.close()

# 5. PHYSICAL MODEL FIT

def plot_model_fit(samples, t_data, y_data, output_dir):

    if t_data is None or y_data is None:
        print("\n✗ Could not plot model fit (data not available)")
        return

    print("\n" + "-"*80)
    print("GENERATING MODEL FIT PLOT")
    print("-"*80)

    def flare_model(t, A, tau, omega):
        """Flare intensity model"""
        return A * np.exp(t) * (1 - np.tanh(2 * (t - tau))) * np.sin(omega * t)

    # Get MAP estimate (mean of posterior)
    map_estimate = np.mean(samples, axis=0)
    A_map, tau_map, omega_map = map_estimate

    # Generate model predictions
    t_fine = np.linspace(t_data.min(), t_data.max(), 500)
    y_model = flare_model(t_fine, A_map, tau_map, omega_map)

    # Generate predictions from multiple posterior samples for uncertainty band
    n_samples_plot = 100
    sample_indices = np.random.choice(len(samples), n_samples_plot, replace=False)
    y_samples = np.array([flare_model(t_fine, *samples[i]) for i in sample_indices])
    y_lower = np.percentile(y_samples, 5, axis=0)
    y_upper = np.percentile(y_samples, 95, axis=0)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Full data
    ax1.scatter(t_data, y_data, alpha=0.4, s=20, label='Observed Data', color='gray')
    ax1.plot(t_fine, y_model, 'r-', linewidth=2.5, label='MAP Model Fit')
    ax1.fill_between(t_fine, y_lower, y_upper, alpha=0.2, color='red',
                     label='90% Posterior Interval')
    ax1.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax1.set_title('Solar Flare Intensity: Observed Data vs Model Fit (Full Range)',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')

    # Zoomed to peak region
    peak_idx = np.argmax(np.abs(y_data))
    peak_time = t_data[peak_idx]
    zoom_range = 0.5
    zoom_mask = (t_data >= peak_time - zoom_range) & (t_data <= peak_time + zoom_range)
    zoom_fine_mask = (t_fine >= peak_time - zoom_range) & (t_fine <= peak_time + zoom_range)

    ax2.scatter(t_data[zoom_mask], y_data[zoom_mask], alpha=0.6, s=30,
               label='Observed Data', color='gray', edgecolors='black')
    ax2.plot(t_fine[zoom_fine_mask], y_model[zoom_fine_mask], 'r-', linewidth=2.5,
            label='MAP Model Fit')
    ax2.fill_between(t_fine[zoom_fine_mask], y_lower[zoom_fine_mask], y_upper[zoom_fine_mask],
                    alpha=0.2, color='red', label='90% Posterior Interval')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax2.set_title(f'Zoomed View: Peak Region (t ~ {peak_time:.2f})',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')

    # Add parameter values as text (avoid unicode for Windows compatibility)
    textstr = f'MAP Estimates:\nA = {A_map:.4f}\ntau = {tau_map:.4f}\nomega = {omega_map:.4f}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    # Save figure
    fit_file = output_dir / 'model_fit.png'
    plt.savefig(fit_file, dpi=300, bbox_inches='tight')
    print(f"✓ Model fit plot saved: {fit_file}")
    plt.close()

# 6. SUMMARY STATISTICS TABLE

def generate_summary_report(samples, output_dir):

    print("\n" + "-"*80)
    print("GENERATING SUMMARY REPORT")
    print("-"*80)

    param_names = ['A (Amplitude)', 'tau (Quench Time)', 'omega (Frequency)']
    param_units = ['dimensionless', 'time units', 'rad/time']

    # Open report file with UTF-8 encoding (FIX for Windows)
    report_file = output_dir / 'posterior_summary.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SOLAR FLARE PULSE: STOCHASTIC SIGNAL RECOVERY\n")
        f.write("Bayesian Parameter Estimation - POSTERIOR STATISTICS\n")
        f.write("="*80 + "\n\n")

        f.write("MCMC RESULTS SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total posterior samples: {samples.shape[0]}\n")
        f.write(f"Number of parameters: {samples.shape[1]}\n\n")

        for i, (name, units) in enumerate(zip(param_names, param_units)):
            param_samples = samples[:, i]

            mean_val = np.mean(param_samples)
            median_val = np.median(param_samples)
            std_val = np.std(param_samples)
            mad_val = stats.median_abs_deviation(param_samples)

            ci_lower_68 = np.percentile(param_samples, 16)
            ci_upper_68 = np.percentile(param_samples, 84)
            ci_lower_95 = np.percentile(param_samples, 2.5)
            ci_upper_95 = np.percentile(param_samples, 97.5)

            f.write(f"\n{'='*80}\n")
            f.write(f"PARAMETER {i+1}: {name} [{units}]\n")
            f.write(f"{'='*80}\n\n")

            f.write("POINT ESTIMATES:\n")
            f.write(f"  Posterior Mean:           {mean_val:12.6f}\n")
            f.write(f"  Posterior Median:         {median_val:12.6f}\n")
            f.write(f"  Posterior Mode (approx):  {np.median(param_samples):12.6f}\n\n")

            f.write("UNCERTAINTY MEASURES:\n")
            f.write(f"  Standard Deviation:       {std_val:12.6f}\n")
            f.write(f"  Median Abs Deviation:     {mad_val:12.6f}\n")
            f.write(f"  Interquartile Range:      {ci_upper_68 - ci_lower_68:12.6f}\n\n")

            f.write("CREDIBLE INTERVALS:\n")
            f.write(f"  68% CI (1 std dev):       [{ci_lower_68:10.6f}, {ci_upper_68:10.6f}]\n")
            f.write(f"  95% CI:                   [{ci_lower_95:10.6f}, {ci_upper_95:10.6f}]\n\n")

            f.write("DISTRIBUTION SUMMARY:\n")
            f.write(f"  Minimum:                  {np.min(param_samples):12.6f}\n")
            f.write(f"  25th percentile:          {np.percentile(param_samples, 25):12.6f}\n")
            f.write(f"  50th percentile (median): {np.percentile(param_samples, 50):12.6f}\n")
            f.write(f"  75th percentile:          {np.percentile(param_samples, 75):12.6f}\n")
            f.write(f"  Maximum:                  {np.max(param_samples):12.6f}\n\n")

            # Skewness and kurtosis
            skewness = stats.skew(param_samples)
            kurtosis = stats.kurtosis(param_samples)
            f.write(f"SHAPE DESCRIPTORS:\n")
            f.write(f"  Skewness:                 {skewness:12.6f}\n")
            f.write(f"  Kurtosis:                 {kurtosis:12.6f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary report saved: {report_file}")


def main():

    print("\n" + "="*80)
    print("SOLAR FLARE MCMC VISUALIZATION & RESULTS GENERATION")
    print("="*80)

    # Load results
    samples, data = load_results()

    if samples is None:
        print("\n✗ Could not load MCMC results. Exiting.")
        return

    t_data, y_data = data if data else (None, None)

    # Generate all plots
    plot_trace_plots(samples, output_dir)
    plot_posterior_distributions(samples, output_dir)
    plot_corner_plot(samples, output_dir)
    plot_convergence_diagnostics(samples, output_dir)
    plot_model_fit(samples, t_data, y_data, output_dir)

    # Generate summary report
    generate_summary_report(samples, output_dir)

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}\n")
    print("Generated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  ✓ {file.name}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("1. View plots in ./output/ folder")
    print("2. Check posterior_summary.txt for detailed statistics")
    print("3. Review trace_plots.png for convergence assessment")
    print("4. Examine posterior_distributions.png for uncertainty quantification")
    print("="*80)


if __name__ == "__main__":
    main()