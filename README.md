"""# Task 2: Data Cleaning & Missing Value Handling
 
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
