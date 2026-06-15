import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Set seed for identical distributions
np.random.seed(42)

# Simulate VAE reconstruction errors (MSE)
# Perfect reps: skewed close to 0
perfect_errors = np.random.gamma(shape=2.5, scale=0.008, size=300) 
# Flawed reps (cheating form, swinging, low range): higher mean, wider variance
flawed_errors = np.random.normal(loc=0.075, scale=0.018, size=200)
flawed_errors = flawed_errors[flawed_errors > 0.025] # clean up edge cases

# Calculate the actual 95th percentile threshold from perfect data
threshold = np.percentile(perfect_errors, 95)

# --- Plotting Canvas Setup ---
plt.figure(figsize=(12, 6.5))

# Compute smooth KDE lines
kde_perfect = stats.gaussian_kde(perfect_errors)
kde_flawed = stats.gaussian_kde(flawed_errors)

x_perf = np.linspace(0, 0.06, 500)
x_flaw = np.linspace(0.02, 0.13, 500)

# Plot curves with clean fills
plt.plot(x_perf, kde_perfect(x_perf), color='#2ECC71', linewidth=3, label='Perfect Repetitions (In-Distribution Training Data)')
plt.fill_between(x_perf, kde_perfect(x_perf), color='#2ECC71', alpha=0.3)

plt.plot(x_flaw, kde_flawed(x_flaw), color='#E74C3C', linewidth=3, label='Flawed / Incorrect Repetitions (Anomalies)')
plt.fill_between(x_flaw, kde_flawed(x_flaw), color='#E74C3C', alpha=0.3)

# Draw the 95th Percentile Decision Threshold line
plt.axvline(threshold, color='#2C3E50', linestyle='--', linewidth=2.5, 
            label=f'Calibrated Threshold (95th Percentile = {threshold:.4f})')

# Add shaded background zones for decision feedback
plt.axvspan(0, threshold, color='#E8F8F5', alpha=0.4, zorder=0)
plt.axvspan(threshold, 0.14, color='#FDEDEC', alpha=0.4, zorder=0)

# Add clear on-graph labeling for the committee
plt.text(threshold - 0.003, plt.gca().get_ylim()[1] * 0.75, 'VALID REPS\n(Accepted Form)', 
         color='#1E8449', fontsize=12, fontweight='bold', ha='right')
plt.text(threshold + 0.003, plt.gca().get_ylim()[1] * 0.75, 'OUT-OF-MANIFOLD\n(Form Anomaly)', 
         color='#922B21', fontsize=12, fontweight='bold', ha='left')

# Graph Details
plt.title('Semi-Supervised Anomaly Detection Manifold', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('VAE Reconstruction Error ($MSE$ between Input $X$ and Reconstruction $\hat{X}$)', fontsize=12, labelpad=10)
plt.ylabel('Probability Density', fontsize=12, labelpad=10)
plt.xlim(0, 0.13)
plt.ylim(0, plt.gca().get_ylim()[1] * 1.05) # Add a tiny bit of headroom
plt.legend(loc='upper right', fontsize=10.5, frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

plt.show()