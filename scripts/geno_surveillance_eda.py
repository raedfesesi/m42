import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV efficiently from the DigitalOcean URL
file_url = 'https://storage.botforce.ai/m42/vwjoingenus2_202505021928.csv'

try:
    df_chunks = pd.read_csv(file_url, chunksize=50000)
    df_list = [chunk for chunk in df_chunks if not chunk.empty]
    if not df_list:
        print("Error: Loaded CSV is empty.")
        exit(1)
    df = pd.concat(df_list, ignore_index=True)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

if df.empty:
    print("Error: Final dataframe is empty after loading.")
    exit(1)

# Step 2: Ensure correct data types
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

for col in ['result_value', 'percentage', 'median']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 3: Basic Overview
print("--- Dataset Overview ---")
print(df.shape)
print(df.dtypes)
print(df.head())

# Step 4: Optimize dtypes (exclude Date)
for col in df.select_dtypes(include='object').columns:
    if col != 'Date' and df[col].nunique() > 0 and df[col].nunique() / df.shape[0] < 0.5:
        df[col] = df[col].astype('category')

# Step 5: Check for Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Step 6: Descriptive Statistics
print("\n--- Descriptive Statistics ---")
print(df.describe(include='all'))

# Step 7: Unique Values per Column
print("\n--- Unique Values per Column ---")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Step 8: Time Coverage
if 'Date' in df.columns and df['Date'].notnull().any():
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    print(f"\nDate Range: {min_date} to {max_date}")
    daily_counts = df.set_index('Date').resample('D').size()
    daily_counts.plot(figsize=(12, 6), title='Records per Day', linewidth=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'Date' column could not be parsed into valid dates.")

# Step 9: Group-level Statistics
group_stats = df.groupby('text_id', observed=False).agg({
    'result_value': ['min', 'max', 'mean', 'std'],
    'percentage': ['min', 'max', 'mean', 'std'],
    'median': ['min', 'max', 'mean', 'std']
}).reset_index()
print("\n--- Group-Level Statistics (per text_id) ---")
print(group_stats)

# Step 10: Correlation Matrix (sample if large)
sample_df = df.sample(n=10000, random_state=42) if df.shape[0] > 10000 else df
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
if not numeric_cols.empty:
    sns.heatmap(sample_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix (Sampled)')
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found for correlation matrix.")

# Step 11: Outlier Detection (both raw and log scale)
for col in ['result_value', 'percentage', 'median']:
    if col in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[col]):
        # Raw boxplot
        sns.boxplot(x=sample_df[col])
        plt.title(f'Boxplot of {col} (Sampled)')
        plt.tight_layout()
        plt.show()
        # Log-scaled boxplot (optional)
        sns.boxplot(x=np.log1p(sample_df[col]))
        plt.title(f'Log Boxplot of {col} (Sampled)')
        plt.tight_layout()
        plt.show()

# Step 12: Seasonal Patterns
if 'Date' in df.columns and df['Date'].notnull().any():
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    monthly_counts = df.groupby(['year', 'month'], observed=False).size().unstack(0)
    monthly_counts.plot(kind='bar', figsize=(12, 6), title='Monthly Record Counts by Year')
    plt.tight_layout()
    plt.show()

print("Exploration completed. Ready for anomaly detection setup.")
