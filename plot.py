import pandas as pd
import matplotlib.pyplot as plt

# 1) List your files and the labels you want
files_and_labels = {
    'synthetic_gap_meanrisk.csv':    'Synthetic (mean-risk)',
    'ff10_gap_meanrisk.csv':          'Fama–French (mean-risk)',
    'synthetic_gap_shortfall.csv':    'Synthetic (shortfall)',
    'ff10_gap_shortfall.csv':         'Fama–French (shortfall)',
}

# 2) Read and plot
plt.figure(figsize=(8,5))
for fname, label in files_and_labels.items():
    df = pd.read_csv(fname)
    plt.plot(df['epsilon'], df['pct_gain'], marker='o', linewidth=1, label=label)

# 3) Styling
plt.xscale('log')
plt.xlabel('Wasserstein radius $\\varepsilon$')
plt.ylabel('Relative gain (%)')
plt.title('DRO vs. SAA: % Gain Across Datasets & Losses')
plt.legend(loc='best')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()

# 4) Show or save
plt.show()
# plt.savefig('combined_performance_gap.png', dpi=300)
