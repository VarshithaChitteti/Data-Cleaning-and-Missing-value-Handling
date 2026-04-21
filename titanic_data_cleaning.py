# ============================================================
# EDUTECH SOLUTION - AI & ML INTERNSHIP
# Task 2: Data Cleaning & Missing Value Handling
# Dataset: Titanic
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
# ============================================================

# ─────────────────────────────────────────────
# CELL 1: Install & Import Libraries
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ All libraries imported successfully!")


# ─────────────────────────────────────────────
# CELL 2: Load the Titanic Dataset
# ─────────────────────────────────────────────
# Option A: Load directly from seaborn (no download needed)
df = sns.load_dataset('titanic')

# Option B (alternative): Load from URL
# url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
# df = pd.read_csv(url)

print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print(df.head())


# ─────────────────────────────────────────────
# CELL 3: Initial Exploration
# ─────────────────────────────────────────────
print("=" * 50)
print("DATASET INFO")
print("=" * 50)
df.info()

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe(include='all'))

print("\n" + "=" * 50)
print("COLUMN DATA TYPES")
print("=" * 50)
print(df.dtypes)


# ─────────────────────────────────────────────
# CELL 4: Identify Missing Values
# ─────────────────────────────────────────────
print("=" * 50)
print("MISSING VALUE COUNT PER COLUMN")
print("=" * 50)
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %': missing_percent.round(2)
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_df)

print(f"\nTotal missing values in dataset: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# CELL 5: Visualize Missing Data — Heatmap
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Value Heatmap (Yellow = Missing)', fontsize=15, fontweight='bold')
plt.xlabel('Columns')
plt.tight_layout()
plt.savefig('missing_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Heatmap saved as 'missing_heatmap.png'")


# ─────────────────────────────────────────────
# CELL 6: Visualize Missing Data — Bar Chart
# ─────────────────────────────────────────────
missing_only = df.isnull().sum()
missing_only = missing_only[missing_only > 0]

plt.figure(figsize=(10, 5))
bars = plt.bar(missing_only.index, missing_only.values, color=['#e74c3c', '#e67e22', '#3498db'])
plt.title('Count of Missing Values per Column', fontsize=14, fontweight='bold')
plt.xlabel('Columns')
plt.ylabel('Missing Count')
for bar, val in zip(bars, missing_only.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('missing_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Bar chart saved as 'missing_bar.png'")


# ─────────────────────────────────────────────
# CELL 7: Handle Missing Values
# ─────────────────────────────────────────────

# --- Keep a copy of raw data for comparison ---
df_raw = df.copy()
df_clean = df.copy()

print("BEFORE CLEANING:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
print()

# ── 7a. Drop the 'deck' column (>77% missing — too much to impute) ──
df_clean.drop(columns=['deck'], inplace=True)
print("✅ Dropped 'deck' column (77% missing — not worth imputing)")

# ── 7b. Drop 'embark_town' and 'embarked' duplicate columns ──
# 'embarked' and 'embark_town' carry the same info — keep 'embarked'
df_clean.drop(columns=['embark_town'], inplace=True)
print("✅ Dropped 'embark_town' (duplicate of 'embarked')")

# ── 7c. Impute 'age' with MEDIAN (robust to outliers) ──
age_imputer = SimpleImputer(strategy='median')
df_clean['age'] = age_imputer.fit_transform(df_clean[['age']])
print(f"✅ Imputed 'age' with Median = {df_raw['age'].median()}")

# ── 7d. Impute 'embarked' with MODE (categorical column) ──
mode_imputer = SimpleImputer(strategy='most_frequent')
df_clean['embarked'] = mode_imputer.fit_transform(df_clean[['embarked']]).ravel()
print(f"✅ Imputed 'embarked' with Mode = '{df_raw['embarked'].mode()[0]}'")

# ── 7e. Drop rows where 'alive' column is missing ──
df_clean.dropna(subset=['alive'], inplace=True)
print(f"✅ Dropped rows with missing 'alive' values")

print("\nAFTER CLEANING:")
remaining = df_clean.isnull().sum()[df_clean.isnull().sum() > 0]
if remaining.empty:
    print("🎉 No missing values remain!")
else:
    print(remaining)


# ─────────────────────────────────────────────
# CELL 8: Outlier Detection & Treatment
# ─────────────────────────────────────────────

# ── 8a. Boxplots BEFORE outlier treatment ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['age', 'fare', 'sibsp']):
    sns.boxplot(y=df_clean[col], ax=ax, color='#3498db')
    ax.set_title(f'{col.upper()} - Before Outlier Treatment', fontweight='bold')
plt.suptitle('Boxplots Before Outlier Treatment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outliers_before.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 'outliers_before.png'")

# ── 8b. IQR-based Outlier Capping (Winsorization) ──
def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"  {col}: {before} outliers capped → [{lower:.2f}, {upper:.2f}]")
    return df

print("\nCapping outliers using IQR method:")
for col in ['age', 'fare', 'sibsp', 'parch']:
    df_clean = cap_outliers_iqr(df_clean, col)

# ── 8c. Boxplots AFTER outlier treatment ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['age', 'fare', 'sibsp']):
    sns.boxplot(y=df_clean[col], ax=ax, color='#2ecc71')
    ax.set_title(f'{col.upper()} - After Outlier Treatment', fontweight='bold')
plt.suptitle('Boxplots After Outlier Treatment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outliers_after.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 'outliers_after.png'")


# ─────────────────────────────────────────────
# CELL 9: Final Verification
# ─────────────────────────────────────────────
print("=" * 50)
print("FINAL DATASET VERIFICATION")
print("=" * 50)
print(f"Original shape : {df_raw.shape}")
print(f"Cleaned shape  : {df_clean.shape}")
print(f"\nMissing values remaining: {df_clean.isnull().sum().sum()}")
print(f"\nDuplicate rows: {df_clean.duplicated().sum()}")

# Drop any duplicate rows
df_clean.drop_duplicates(inplace=True)
print(f"Duplicates after removal: {df_clean.duplicated().sum()}")

print(f"\n✅ FINAL SHAPE: {df_clean.shape}")
print("\nCleaned Dataset Sample:")
print(df_clean.head(10))


# ─────────────────────────────────────────────
# CELL 10: Post-Cleaning Heatmap
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 6))
sns.heatmap(df_clean.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Value Heatmap AFTER Cleaning (All Clean = No Yellow)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('final_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 'final_heatmap.png'")


# ─────────────────────────────────────────────
# CELL 11: Distribution Comparison (Before vs After)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_raw['age'].dropna(), bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[0].set_title('Age Distribution - BEFORE Cleaning', fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

axes[1].hist(df_clean['age'], bins=30, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[1].set_title('Age Distribution - AFTER Cleaning', fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')

plt.suptitle('Age Column: Before vs After Imputation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('age_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 'age_comparison.png'")


# ─────────────────────────────────────────────
# CELL 12: Correlation Heatmap on Cleaned Data
# ─────────────────────────────────────────────
numeric_df = df_clean.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Matrix — Cleaned Titanic Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 'correlation_heatmap.png'")


# ─────────────────────────────────────────────
# CELL 13: Export Cleaned CSV
# ─────────────────────────────────────────────
df_clean.to_csv('titanic_cleaned.csv', index=False)
print("✅ Cleaned dataset exported as 'titanic_cleaned.csv'")
print(f"Final rows: {df_clean.shape[0]}, Final columns: {df_clean.shape[1]}")

# For Google Colab — auto-download the file
try:
    from google.colab import files
    files.download('titanic_cleaned.csv')
    print("✅ File downloaded to your computer!")
except:
    print("ℹ️  Not running in Colab. File saved to current directory.")


# ─────────────────────────────────────────────
# CELL 14: Interview Q&A Summary (Print)
# ─────────────────────────────────────────────
summary = """
╔══════════════════════════════════════════════════════════════╗
║          INTERVIEW QUESTIONS — QUICK REFERENCE               ║
╠══════════════════════════════════════════════════════════════╣
║ Q1: Types of Missing Data?                                   ║
║   • MCAR – Missing Completely At Random (no pattern)         ║
║   • MAR  – Missing At Random (depends on other columns)      ║
║   • MNAR – Missing Not At Random (depends on missing value)  ║
╠══════════════════════════════════════════════════════════════╣
║ Q2: Drop rows vs Impute?                                     ║
║   • Drop   → < 5% missing OR MCAR                           ║
║   • Impute → > 5% OR important feature                      ║
╠══════════════════════════════════════════════════════════════╣
║ Q3: Outliers & Imputation?                                   ║
║   • Outliers skew Mean → prefer Median for skewed data       ║
║   • Remove/cap outliers BEFORE mean imputation               ║
╠══════════════════════════════════════════════════════════════╣
║ Q4: Mean vs Median Imputation?                               ║
║   • Mean   → symmetric distributions, no outliers           ║
║   • Median → skewed data or data with outliers               ║
╠══════════════════════════════════════════════════════════════╣
║ Q5: Role of Data Cleaning in ML Pipeline?                    ║
║   • Garbage in = Garbage out                                 ║
║   • Clean data → better model accuracy & reliability        ║
╚══════════════════════════════════════════════════════════════╝
"""
print(summary)


# ─────────────────────────────────────────────
# CELL 15: README Generator (for GitHub)
# ─────────────────────────────────────────────
readme = """# Task 2: Data Cleaning & Missing Value Handling

## Objective
Clean the Titanic dataset by handling missing values, outliers, and verifying data quality.

## Dataset
- **Source**: Seaborn's built-in Titanic dataset (891 rows × 15 columns)

## Tools Used
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (SimpleImputer)
- SciPy

## Steps Performed
1. **Loaded** dataset and explored shape, dtypes, and statistics
2. **Identified** missing values using `.isnull().sum()`
3. **Visualized** missing data using heatmaps and bar charts
4. **Handled** missing values:
   - Dropped `deck` column (>77% missing)
   - Imputed `age` with **Median** using SimpleImputer
   - Imputed `embarked` with **Mode** using SimpleImputer
   - Dropped rows with missing `alive` values
5. **Detected and capped outliers** in `age`, `fare`, `sibsp`, `parch` using IQR method
6. **Verified** final dataset — 0 missing values
7. **Exported** cleaned CSV

## Output Files
- `titanic_cleaned.csv` — Final cleaned dataset
- `missing_heatmap.png` — Before cleaning heatmap
- `final_heatmap.png` — After cleaning heatmap
- `outliers_before.png` / `outliers_after.png` — Outlier treatment
- `age_comparison.png` — Distribution before vs after
- `correlation_heatmap.png` — Feature correlations

## Key Learnings
- MCAR / MAR / MNAR types of missing data
- When to drop vs impute
- IQR-based outlier capping (Winsorization)
- Mean vs Median imputation tradeoffs
"""

with open('README.md', 'w') as f:
    f.write(readme)
print("✅ README.md generated!")
print("\n🎉 ALL DONE! Your project is complete.")
print("Upload the following to GitHub:")
print("  - titanic_data_cleaning.py (this file)")
print("  - titanic_cleaned.csv")
print("  - All .png images")
print("  - README.md")
