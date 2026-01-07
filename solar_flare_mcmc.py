import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# SECTION 1: PHYSICAL MODEL & LIKELIHOOD

def flare_model(t, A, tau, omega):

    exp_growth = np.exp(t)
    quench = 1 - np.tanh(2 * (t - tau))
    oscillation = np.sin(omega * t)

    return A * exp_growth * quench * oscillation


def log_likelihood(theta, t_data, y_data):

    A, tau, omega = theta

    # Model prediction
    y_model = flare_model(t_data, A, tau, omega)

    # Residuals
    residuals = y_data - y_model

    # Error model with floor for stability
    sigma = np.maximum(0.2 * np.abs(y_data), 1e-10)

    # Chi-squared statistic
    chi2 = np.sum((residuals / sigma) ** 2)

    return -0.5 * chi2


def log_prior(theta):
    A, tau, omega = theta

    if not (0 < A < 2 and 1 < tau < 10 and 1 < omega < 20):
        return -np.inf

    return 0.0


def log_posterior(theta, t_data, y_data):

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, t_data, y_data)


# SECTION 2: ADAPTIVE METROPOLIS-HASTINGS MCMC SAMPLER
class AdaptiveMetropolisHastings:

    def __init__(self, log_posterior, initial_theta, data_args,
                 proposal_sd=None, adapt_rate=0.05):

        self.log_posterior = log_posterior
        self.theta = np.array(initial_theta, dtype=float)
        self.data_args = data_args
        self.dim = len(self.theta)

        # Initialize proposal covariance
        if proposal_sd is None:
            proposal_sd = np.array([0.05, 0.2, 1.0])
        self.proposal_cov = np.diag(proposal_sd ** 2)

        self.adapt_rate = adapt_rate
        self.iteration = 0
        self.n_accepted = 0

        # Storage for diagnostics
        self.samples = []
        self.acceptance_rate = []
        self.log_post_values = []

        # Evaluate at starting point
        self.current_lp = log_posterior(self.theta, *data_args)

    def step(self):

        # Proposal step
        proposal = np.random.multivariate_normal(
            self.theta, self.proposal_cov
        )

        # Evaluate posterior at proposal
        proposed_lp = self.log_posterior(proposal, *self.data_args)

        # Log acceptance ratio
        log_alpha = proposed_lp - self.current_lp

        # Accept/reject decision
        accepted = np.log(np.random.uniform()) < log_alpha

        if accepted:
            self.theta = proposal
            self.current_lp = proposed_lp
            self.n_accepted += 1

        self.iteration += 1
        return accepted

    def adapt_proposal(self, batch_size=50):

        if len(self.samples) < batch_size:
            return

        # Recent samples
        recent_samples = np.array(self.samples[-batch_size:])
        empirical_cov = np.cov(recent_samples.T)

        # Handle 1D case
        if self.dim == 1:
            empirical_cov = np.array([[empirical_cov]])

        # Blend factor: higher early, lower later
        weight = self.adapt_rate

        # Exponential smoothing blend
        self.proposal_cov = (
            (1 - weight) * self.proposal_cov +
            weight * empirical_cov
        )

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(self.proposal_cov)
        if np.min(eigvals) < 1e-6:
            self.proposal_cov += 1e-6 * np.eye(self.dim)

    def run(self, n_iterations, adapt_interval=50, verbose=True):

        pbar = tqdm(range(n_iterations), disable=not verbose)

        for i in pbar:
            # MCMC step
            self.step()

            # Record samples and diagnostics
            self.samples.append(self.theta.copy())
            self.log_post_values.append(self.current_lp)

            # Running acceptance rate
            acc_rate = self.n_accepted / (i + 1)
            self.acceptance_rate.append(acc_rate)

            # Adaptive proposal update
            if (i + 1) % adapt_interval == 0:
                self.adapt_proposal(batch_size=min(50, (i+1)//10))

                if verbose:
                    pbar.set_postfix({
                        'AR': f'{acc_rate:.3f}',
                        'LP': f'{self.current_lp:.1f}'
                    })

    def get_samples(self):
        """Return all samples as numpy array"""
        return np.array(self.samples)

    def get_acceptance_rate(self):
        """Return overall acceptance rate"""
        return self.n_accepted / len(self.samples)


# SECTION 3: CONVERGENCE DIAGNOSTICS

def gelman_rubin_diagnostic(chains):

    n_chains = len(chains)
    n_iter = chains[0].shape[0]
    n_params = chains[0].shape[1] if chains[0].ndim > 1 else 1

    # Within-chain variance
    W = np.mean([np.var(chain, axis=0, ddof=1) for chain in chains], axis=0)

    # Between-chain variance
    chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
    overall_mean = np.mean(chain_means, axis=0)
    B = n_iter * np.var(chain_means, axis=0, ddof=1)

    # Estimated posterior variance
    Var_hat = ((n_iter - 1) / n_iter) * W + (1 / n_iter) * B

    # R̂ statistic
    R_hat = np.sqrt(Var_hat / W)

    return R_hat


def autocorrelation_function(samples, max_lag=100):

    if samples.ndim == 2:
        samples = samples[:, 0]

    N = len(samples)
    max_lag = min(max_lag, N // 2)  # FIX: Ensure lag doesn't exceed N/2

    samples = samples - np.mean(samples)
    c0 = np.dot(samples, samples) / N

    if c0 == 0:
        return np.ones(max_lag)

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf[lag] = 1.0
        else:
            c_lag = np.dot(samples[:-lag], samples[lag:]) / N
            acf[lag] = c_lag / c0

    return acf


def effective_sample_size(samples):

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    N = samples.shape[0]
    n_params = samples.shape[1]
    ess = np.zeros(n_params)

    for p in range(n_params):
        max_lag_param = min(100, N // 2)
        acf = autocorrelation_function(samples[:, p], max_lag=max_lag_param)

        # Compute integrated autocorrelation time
        # Stop when ACF becomes negative (good practice)
        tau_int = 0.5  # Start with 0.5 (accounting for lag 0 = 1.0)
        for k in range(1, len(acf)):
            if acf[k] > 0:
                tau_int += acf[k]
            else:
                break

        ess[p] = N / (2 * tau_int)

    return ess

# SECTION 4: MAIN EXECUTION
def main():
    """Execute complete Bayesian parameter estimation"""

    # Load data
    print("="*80)
    print("SOLAR FLARE PULSE: STOCHASTIC SIGNAL RECOVERY")
    print("Bayesian Parameter Estimation via Adaptive MCMC")
    print("="*80)

    data = pd.read_csv('flare_data.csv')
    t_data = data['t'].values
    y_data = data['s'].values

    print(f"\n✓ Data loaded: {len(t_data)} observations")
    print(f"  Time range: [{t_data.min():.4f}, {t_data.max():.4f}]")
    print(f"  Signal range: [{y_data.min():.2f}, {y_data.max():.2f}]")

    # Initial parameter estimation
    print("\n" + "-"*80)
    print("Step 1: Finding initial parameter estimates via optimization...")
    print("-"*80)

    def neg_log_post(theta):
        val = -log_posterior(theta, t_data, y_data)
        return np.inf if not np.isfinite(val) else val

    best_theta = None
    best_val = np.inf

    # Multiple random starts for robustness
    for trial in range(5):
        theta0 = np.array([
            np.random.uniform(0.5, 1.5),
            np.random.uniform(3.0, 6.0),
            np.random.uniform(5.0, 15.0)
        ])

        try:
            res = minimize(neg_log_post, theta0, method='L-BFGS-B',
                          bounds=[(0.01, 1.99), (1.01, 9.99), (1.01, 19.99)],
                          options={'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x
        except:
            continue

    if best_theta is None:
        best_theta = np.array([0.8, 5.0, 10.0])

    print(f"Initial parameters: A={best_theta[0]:.4f}, τ={best_theta[1]:.4f}, ω={best_theta[2]:.4f}")

    # Multi-chain MCMC
    print("\n" + "-"*80)
    print("Step 2: Running multiple MCMC chains for convergence assessment...")
    print("-"*80)

    n_chains = 3
    chains = []
    samplers = []

    for chain_id in range(n_chains):
        print(f"\nChain {chain_id + 1}/{n_chains}")

        # Perturbed starting point
        theta_start = best_theta + np.random.normal(0, 0.1, 3)

        # Create sampler
        sampler = AdaptiveMetropolisHastings(
            log_posterior, theta_start, (t_data, y_data),
            proposal_sd=np.array([0.05, 0.2, 1.0]),
            adapt_rate=0.05
        )

        # Burn-in phase (long initial phase)
        print("  Burn-in phase: 3000 iterations")
        sampler.run(3000, adapt_interval=30, verbose=False)
        burn_in_samples = len(sampler.samples)

        # Main sampling phase
        print("  Main phase:    5000 iterations")
        sampler.run(5000, adapt_interval=50, verbose=False)

        # Extract post-burn-in samples
        chain_samples = sampler.get_samples()[burn_in_samples:]
        chains.append(chain_samples)
        samplers.append(sampler)

        print(f"  Acceptance rate: {sampler.get_acceptance_rate():.3f}")

    # Combine all chains
    all_samples = np.vstack(chains)

    print(f"\n✓ Total samples collected: {len(all_samples)}")
    print(f"  Per-chain: {len(chains[0])} samples")

    # Convergence assessment
    print("\n" + "="*80)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*80)

    R_hat = gelman_rubin_diagnostic(chains)
    print("\nGelman-Rubin Diagnostic R̂ (< 1.05 is ideal):")
    param_names = ['A (Amplitude)', 'τ (Quench Time)', 'ω (Frequency)']
    for i, name in enumerate(param_names):
        status = "✓ PASS" if R_hat[i] < 1.1 else "✗ WARNING"
        print(f"  {name:20s}: {R_hat[i]:.6f}  {status}")

    # ESS analysis
    ess = effective_sample_size(all_samples)
    print(f"\nEffective Sample Size (ESS):")
    for i, name in enumerate(param_names):
        ratio = ess[i] / len(all_samples)
        print(f"  {name:20s}: {ess[i]:6.0f} ({ratio:.1%} of total)")

    # Posterior statistics
    print("\n" + "="*80)
    print("POSTERIOR STATISTICS")
    print("="*80)

    # MAP estimate
    max_idx = np.argmax([s.current_lp for s in samplers])
    map_estimate = samplers[max_idx].theta

    # Summary statistics
    post_mean = np.mean(all_samples, axis=0)
    post_median = np.median(all_samples, axis=0)
    post_std = np.std(all_samples, axis=0)
    post_mad = stats.median_abs_deviation(all_samples, axis=0)

    # Credible intervals
    ci_lower_95 = np.percentile(all_samples, 2.5, axis=0)
    ci_upper_95 = np.percentile(all_samples, 97.5, axis=0)
    ci_lower_68 = np.percentile(all_samples, 16, axis=0)
    ci_upper_68 = np.percentile(all_samples, 84, axis=0)

    print("\nParameter Estimates with Credible Intervals:")
    print("="*80)

    for i, name in enumerate(param_names):
        print(f"\n{name}:")
        print(f"  MAP estimate (max. a posteriori):     {map_estimate[i]:.6f}")
        print(f"  Posterior mean:                       {post_mean[i]:.6f}")
        print(f"  Posterior median:                     {post_median[i]:.6f}")
        print(f"  Posterior std dev:                    {post_std[i]:.6f}")
        print(f"  68% credible interval (1σ):           [{ci_lower_68[i]:.6f}, {ci_upper_68[i]:.6f}]")
        print(f"  95% credible interval:                [{ci_lower_95[i]:.6f}, {ci_upper_95[i]:.6f}]")

    # Return results
    results = {
        'samples': all_samples,
        'chains': chains,
        'samplers': samplers,
        'post_mean': post_mean,
        'post_median': post_median,
        'post_std': post_std,
        'map_estimate': map_estimate,
        'ci_95': (ci_lower_95, ci_upper_95),
        'R_hat': R_hat,
        'ESS': ess,
        't_data': t_data,
        'y_data': y_data
    }

    return results


if __name__ == "__main__":
    results = main()
    print("\n" + "="*80)
    print("✓ MCMC Analysis Complete!")
    print("="*80)

    # Save results
    np.save('mcmc_samples.npy', results['samples'])
    print("✓ Samples saved to mcmc_samples.npy")