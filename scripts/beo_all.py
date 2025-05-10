import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')

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

# Step 8: Group-Level Statistics
group_stats = df.groupby('text_id').agg({
    'result_value': ['min', 'max', 'mean', 'std'],
    'percentage': ['min', 'max', 'mean', 'std'],
    'median': ['min', 'max', 'mean', 'std']
}).reset_index()
print("\n--- Group-Level Statistics (per text_id) ---")
print(group_stats.head())

# Step 9: Anomaly Detection
print("\n--- Running Isolation Forest Anomaly Detection ---")
features = ['result_value', 'percentage', 'median']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
df['is_anomaly'] = df['anomaly_score'] == -1

anomaly_count = df['is_anomaly'].sum()
print(f"Detected {anomaly_count} anomalies out of {len(df)} records.")

print("\n--- Sample Anomalies ---")
print(df[df['is_anomaly']].head())

# Step 10: Visualization of Anomalies
print("\n--- Generating Anomaly Visualizations ---")

plt.figure(figsize=(12, 6))
plt.scatter(df['sample_date'], df['result_value'], c=df['is_anomaly'], cmap='coolwarm', s=10, alpha=0.6)
plt.xlabel('Sample Date')
plt.ylabel('Result Value')
plt.title('Anomaly Detection on Result Values Over Time')
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(df['sample_date'], df['percentage'], c=df['is_anomaly'], cmap='coolwarm', s=10, alpha=0.6)
plt.xlabel('Sample Date')
plt.ylabel('Percentage')
plt.title('Anomaly Detection on Percentage Over Time')
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(df['sample_date'], df['median'], c=df['is_anomaly'], cmap='coolwarm', s=10, alpha=0.6)
plt.xlabel('Sample Date')
plt.ylabel('Median')
plt.title('Anomaly Detection on Median Over Time')
plt.show()

# Step 11: Export Anomalies to CSV
output_path = 'beo_anomaly_results.csv'
df[df['is_anomaly']].to_csv(output_path, index=False)
print(f"Anomaly results saved to {output_path}.")

print("\nBeo Surveillance EDA + Anomaly Detection + Visualization completed.")