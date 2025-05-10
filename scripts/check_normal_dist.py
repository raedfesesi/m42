import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your data (if not already loaded)
df = pd.read_csv('https://storage.botforce.ai/m42/vwjoingenus2_202505022026.csv')

# Choose column to test
col = 'median'

# 1️⃣ Visual inspection
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.histplot(df[col], bins=50, kde=True)
plt.title(f'Histogram of {col}')

plt.subplot(1,3,2)
sns.boxplot(x=df[col])
plt.title(f'Boxplot of {col}')

plt.subplot(1,3,3)
stats.probplot(df[col], dist="norm", plot=plt)
plt.title(f'Q-Q Plot of {col}')

plt.tight_layout()
plt.show()

# 2️⃣ Numerical skewness and kurtosis
skewness = stats.skew(df[col].dropna())
kurtosis = stats.kurtosis(df[col].dropna())
print(f"\n{col} Skewness: {skewness:.2f}")
print(f"{col} Kurtosis: {kurtosis:.2f}")

# 3️⃣ Statistical normality tests
shapiro_stat, shapiro_p = stats.shapiro(df[col].sample(5000))  # sample for large data
print(f"Shapiro-Wilk test p-value: {shapiro_p:.5f}")

dagostino_stat, dagostino_p = stats.normaltest(df[col].dropna())
print(f"D’Agostino’s K-squared test p-value: {dagostino_p:.5f}")
