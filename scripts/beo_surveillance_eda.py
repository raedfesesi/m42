Great! Here’s the **full updated Python script** renamed as `beo_surveillance_eda.py` with both exploratory data analysis and anomaly detection integrated.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Step 1: Load CSV
file_url = 'https://storage.botforce.ai/m42/vwjoingenus2_202505022026.csv'

try:
    df_chunks = pd.read_csv(file_url, chunksize=50000)
    df = pd.concat([chunk for chunk in df_chunks if not chunk.empty], ignore_index=True)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

if df.empty:
    print("Error: DataFrame is empty after loading.")
    exit(1)

# Step 2: Fix dtypes
if 'sample_date' in df.columns:
    df['sample_date'] = pd.to_datetime(df['sample_date'], format='%Y-%m-%d', errors='coerce')

for col in ['result_value', 'percentage', 'median']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 3: Overview
print("--- Dataset Overview ---")
print(df.shape)
print(df.dtypes)
print(df.head())

# Step 4: Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Step 5: Descriptive Stats
print("\n--- Descriptive Statistics ---")
print(df.describe(include='all'))

# Step 6: Unique Counts
print("\n--- Unique Values per Column ---")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Step 7: Date Range
if 'sample_date' in df.columns and df['sample_date'].notnull().any():
    min_date = df['sample_date'].min()
    max_date = df['sample_date'].max()
    print(f"\nDate Range: {min_date} to {max_date}")
    df.set_index('sample_date').resample('D').size().plot(figsize=(12, 6), title='Records per Day', linewidth=0.7)
    plt.tight_layout()
    plt.show()

# Step 8: Correlation Matrix
sample_df = df.sample(n=10000, random_state=42) if df.shape[0] > 10000 else df
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
sns.heatmap(sample_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Sampled)')
plt.tight_layout()
plt.show()

# Step 9: Boxplots
for col in ['result_value', 'percentage', 'median']:
    sns.boxplot(x=sample_df[col])
    plt.title(f'Boxplot of {col} (Sampled)')
    plt.tight_layout()
    plt.show()
    sns.boxplot(x=np.log1p(sample_df[col]))
    plt.title(f'Log Boxplot of {col} (Sampled)')
    plt.tight_layout()
    plt.show()

# Step 10: Seasonal Counts
if 'sample_date' in df.columns:
    df['month'] = df['sample_date'].dt.month
    df['year'] = df['sample_date'].dt.year
    monthly_counts = df.groupby(['month', 'year']).size().unstack()
    monthly_counts.plot(kind='bar', figsize=(12, 6), title='Monthly Record Counts by Year')
    plt.tight_layout()
    plt.show()

# Step 11: Anomaly Detection
print("\n--- Running Isolation Forest Anomaly Detection ---")
features = ['result_value', 'percentage', 'median']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
df['is_anomaly'] = df['anomaly_score'] == -1

print(f"Detected {df['is_anomaly'].sum()} anomalies out of {len(df)} records.")
print(df[df['is_anomaly']].head())

if 'sample_date' in df.columns:
    df.set_index('sample_date').groupby('is_anomaly').resample('M').size().unstack(0).plot(kind='bar', figsize=(12, 6), title='Monthly Anomaly Counts')
    plt.tight_layout()
    plt.show()

print("Exploration and anomaly detection completed.")