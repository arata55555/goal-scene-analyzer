import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Cluster 0': [450, 480],
    'Cluster 1': [300, 320],
    'Cluster 2': [150, 160]
}

index_labels = ['Before', 'After']
df_cross = pd.DataFrame(data, index=index_labels)
print("Cross-tabulation DataFrame:")
print(df_cross)

x2, p, dof, expected = stats.chi2_contingency(df_cross, correction=False)

print(f"\nChi-squared Test Results:\nX2: {x2}, p-value: {p}, dof: {dof}")
if p < 0.05:
    print("The difference is statistically significant (p < 0.05).")
else:
    print("The difference is not statistically significant (p >= 0.05).")

n = df_cross.sum().sum()
min_dim = min(df_cross.shape) - 1
cramers_v = np.sqrt(x2 / (n * min_dim))
print(f"Cramér's V: {cramers_v}")
if cramers_v < 0.1:
    interpretation = "weak association"
elif cramers_v < 0.3:
    interpretation = "moderate association"
else:
    interpretation = "strong association"
print(f"Interpretation of Cramér's V: {interpretation}")

def get_adjusted_residuals(observed, expected):
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    n = observed.sum().sum()

    adj_residuals = np.zeros_like(observed, dtype=float)
    rows, cols = observed.shape

    for i in range(rows):
        for j in range(cols):
            diff = observed.iloc[i, j] - expected[i, j]
            p_row = row_totals.iloc[i] / n
            p_col = col_totals.iloc[j] / n
            denom = np.sqrt(expected[i, j] * (1 - p_row) * (1 - p_col))
            adj_residuals[i, j] = diff / denom if denom != 0 else 0

    return pd.DataFrame(adj_residuals, index=observed.index, columns=observed.columns)

adj_res_df = get_adjusted_residuals(df_cross, expected)

print("\nAdjusted Residuals:")
print(adj_res_df.round(2))

plt.figure(figsize=(10, 5))

sns.heatmap(adj_res_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            linewidths=1, linecolor='gray', vmin=-3, vmax=3)
plt.title("Heatmap of Adjusted Residuals", fontsize=14)
plt.xlabel("Clusters")
plt.ylabel("Condition")

plt.show()