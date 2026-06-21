
# Exploratory Data Analysis (EDA) and Machine Learning on Agricultural Yield Dataset
# Week 3 Assignment 1
# Part A: Understanding the Dataset

# Q1. Dataset Overview
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('agriculture_yield_dataset.csv')

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

print("First 10 records:")
print(df.head(10))
# Observations: The dataset has 1500 rows and 8 columns. The columns are `rainfall_mm`, `temperature_c`, `fertilizer_kg`, `irrigation_hours`, `soil_ph` (numerical features), `crop_type`, `soil_type` (categorical features), and `yield_ton_per_hectare` (the target variable).

# Q2. Data Types and Missing Values

print("Data types of each column:")
print(df.dtypes)

print("Missing values per column:")
missing = df.isnull().sum()
print(missing)
print("\nTotal missing values in dataset:", missing.sum())

# Observations: All numerical columns (`rainfall_mm`, `temperature_c`, `fertilizer_kg`, `irrigation_hours`, `soil_ph`, `yield_ton_per_hectare`) are of type `float64`. The two categorical columns (`crop_type`, `soil_type`) are of type `object`. There are no missing values in any column — the dataset is complete.


# Q3. Descriptive Statistics


summary = df.describe()
print(summary)


means = df.describe().loc['mean']
stds = df.describe().loc['std']

print("Feature with highest mean value:",
      means.idxmax(), "->", round(means.max(), 2))
print("Feature with highest standard deviation:",
      stds.idxmax(), "->", round(stds.max(), 2))

# Observations: `rainfall_mm` has the highest mean value (~754 mm), which makes sense since it's measured in millimeters and naturally takes large values compared to other features. `rainfall_mm` also has the highest standard deviation (~255), indicating rainfall varies the most across records — far more spread out than temperature, fertilizer use, irrigation hours, or soil pH.

# Part B: Exploratory Data Analysis (EDA)

# Q4. Distribution Analysis


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(df['rainfall_mm'], bins=30,
                color='steelblue', edgecolor='black')
axes[0, 0].set_title('Rainfall (mm)')
axes[0, 0].set_xlabel('rainfall_mm')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df['temperature_c'], bins=30,
                color='darkorange', edgecolor='black')
axes[0, 1].set_title('Temperature (C)')
axes[0, 1].set_xlabel('temperature_c')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(df['fertilizer_kg'], bins=30,
                color='seagreen', edgecolor='black')
axes[1, 0].set_title('Fertilizer (kg)')
axes[1, 0].set_xlabel('fertilizer_kg')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df['yield_ton_per_hectare'], bins=30,
                color='indianred', edgecolor='black')
axes[1, 1].set_title('Yield (ton/hectare)')
axes[1, 1].set_xlabel('yield_ton_per_hectare')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Observations:
#
# rainfall_mm:
# - Roughly uniform/flat distribution spread between ~300 and ~1200 mm, with no strong central peak.
# - No extreme outliers — values stay within a bounded, realistic range.
# - Suggests rainfall was sampled fairly evenly across its range in this dataset rather than following a natural bell curve.
#
# temperature_c:
# - Also fairly uniform between ~18°C and ~38°C.
# - No single dominant temperature range — the field conditions cover a wide span of climates/seasons.
# - No visible outliers or gaps.
#
# fertilizer_kg:
# - Spread broadly across the observed range with a roughly flat/uniform shape, similar to rainfall and temperature.
# - No major skew or long tail.
# - Indicates fertilizer application amounts were varied systematically across samples.
#
# yield_ton_per_hectare:
# - Unlike the other three, this one shows a clear bell-shaped, roughly symmetric distribution centered around 5 ton/hectare.
# - Most values fall between ~3.5 and ~6.5 ton/hectare.
# - A few low and high values taper off at the edges, but nothing extreme — consistent with yield being an *outcome* influenced by the (more uniformly distributed) input features.

# Q5. Crop Type Analysis


crop_counts = df['crop_type'].value_counts()
print("Record count per crop type:")
print(crop_counts)


plt.figure(figsize=(7, 5))
crop_counts.plot(kind='bar', color='mediumseagreen', edgecolor='black')
plt.title('Count of Records by Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


print("Most frequent crop type:", crop_counts.idxmax(),
      "with", crop_counts.max(), "records")

# Observation: Cotton is the most frequent crop type (311 records), closely followed by Soybean, Wheat, Rice and Maize — the five crop types are fairly balanced, each making up roughly 19–21% of the dataset.


# Q6. Soil Type Analysis


soil_counts = df['soil_type'].value_counts()
print("Frequency of each soil type:")
print(soil_counts)


plt.figure(figsize=(6, 5))
soil_counts.plot(kind='bar', color='peru', edgecolor='black')
plt.title('Count of Records by Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


print("Most common soil type:", soil_counts.idxmax(),
      "with", soil_counts.max(), "records")


# Observation: Clay is the most common soil type (534 records), followed by Sandy (492) and Loamy (474). The three soil types are reasonably balanced, with Clay only moderately ahead.

# Q7. Yield Distribution


plt.figure(figsize=(8, 5))
plt.hist(df['yield_ton_per_hectare'], bins=30,
         color='cornflowerblue', edgecolor='black')
plt.title('Distribution of Yield (ton/hectare)')
plt.xlabel('yield_ton_per_hectare')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


skewness = df['yield_ton_per_hectare'].skew()
print("Skewness of yield:", round(skewness, 3))

plt.figure(figsize=(6, 4))
plt.boxplot(df['yield_ton_per_hectare'], vert=False)
plt.title('Boxplot of Yield (ton/hectare) - Outlier Check')
plt.xlabel('yield_ton_per_hectare')
plt.tight_layout()
plt.show()

Q1 = df['yield_ton_per_hectare'].quantile(0.25)
Q3 = df['yield_ton_per_hectare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = df[(df['yield_ton_per_hectare'] < lower) |
              (df['yield_ton_per_hectare'] > upper)]
print("Number of outliers (IQR method):", len(outliers))


# Observations:
# - The yield distribution is approximately normal (bell-shaped) and roughly symmetric, with a skewness value close to 0, confirming it doesn't lean heavily to either side.
# - The boxplot and IQR check show very few to no significant outliers — the data is well-behaved and tightly clustered around the mean (~5.03 ton/hectare) without extreme values pulling the distribution.


# Q8. Scatter Plot Analysis


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(df['rainfall_mm'], df['yield_ton_per_hectare'],
                alpha=0.4, color='teal')
axes[0].set_title('Rainfall vs Yield')
axes[0].set_xlabel('rainfall_mm')
axes[0].set_ylabel('yield_ton_per_hectare')

axes[1].scatter(df['fertilizer_kg'], df['yield_ton_per_hectare'],
                alpha=0.4, color='firebrick')
axes[1].set_title('Fertilizer vs Yield')
axes[1].set_xlabel('fertilizer_kg')
axes[1].set_ylabel('yield_ton_per_hectare')

plt.tight_layout()
plt.show()

corr_rainfall = df['rainfall_mm'].corr(df['yield_ton_per_hectare'])
corr_fertilizer = df['fertilizer_kg'].corr(df['yield_ton_per_hectare'])
print("Correlation - rainfall_mm vs yield:", round(corr_rainfall, 3))
print("Correlation - fertilizer_kg vs yield:", round(corr_fertilizer, 3))

# Observation: `rainfall_mm` shows a stronger, clearer positive relationship with yield (correlation ≈ 0.55) compared to `fertilizer_kg` (correlation ≈ 0.28), whose scatter plot looks far more spread out and weakly related. So rainfall has the stronger relationship with yield between these two features.


# Q9. Correlation Analysis


corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()


top3 = corr_matrix['yield_ton_per_hectare'].drop(
    'yield_ton_per_hectare').abs().sort_values(ascending=False).head(3)
print("Top 3 features most correlated with yield_ton_per_hectare:")
print(top3)


# Observation: The top three features most correlated with crop yield are rainfall_mm (≈0.55), irrigation_hours (≈0.54), and fertilizer_kg (≈0.28) — all positively correlated, meaning more water and fertilizer input is associated with higher yield. `soil_ph` and `temperature_c` show almost no linear relationship with yield.


# Q10. Group-Based Analysis


avg_yield_crop = df.groupby('crop_type')[
    'yield_ton_per_hectare'].mean().sort_values(ascending=False)
print("Average yield by crop type:")
print(avg_yield_crop)


avg_yield_soil = df.groupby('soil_type')[
    'yield_ton_per_hectare'].mean().sort_values(ascending=False)
print("Average yield by soil type:")
print(avg_yield_soil)


print("Crop type with highest average yield:",
      avg_yield_crop.idxmax(), "->", round(avg_yield_crop.max(), 3))
print("Soil type with highest average yield:",
      avg_yield_soil.idxmax(), "->", round(avg_yield_soil.max(), 3))


# Observation: Average yield is fairly close across crop and soil types in this dataset, but the crop type and soil type with the highest average yield are printed above based on the actual computed values.


# Part C: Data Preparation


# Q11. Feature Encoding


categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", categorical_cols)


df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
print("Shape after One-Hot Encoding:", df_encoded.shape)
print(df_encoded.head())


# Observation: The categorical columns `crop_type` and `soil_type` were converted into numerical (binary 0/1) columns using One-Hot Encoding via `pd.get_dummies()`. Each unique category becomes its own column.


# Q12. Feature Selection


X = df_encoded.drop('yield_ton_per_hectare', axis=1)
y = df_encoded['yield_ton_per_hectare']

print("Target variable (y): yield_ton_per_hectare")
print("\nInput features (X) columns:")
print(X.columns.tolist())
print("\nX shape:", X.shape)
print("y shape:", y.shape)


# Observation: `yield_ton_per_hectare` is used as the target variable (y), since it's the outcome we want to predict. All remaining columns (numerical features + one-hot encoded categorical columns) form the input features (X).

# Part D: Machine Learning

# Q13. Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Observation: The dataset (1500 rows) was split into 80% training (1200 rows) and 20% testing (300 rows), using `random_state=42` for reproducibility.

# Q14. Linear Regression Model


model = LinearRegression()
model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Intercept:", model.intercept_)
print("\nModel coefficients:")
print(coef_df.to_string(index=False))


highest_pos = coef_df.iloc[0]
print(
    f"Feature with the highest positive coefficient: {highest_pos['Feature']} ({highest_pos['Coefficient']:.4f})")


y_pred = model.predict(X_test)
print("R2 score on test set:", round(r2_score(y_test, y_pred), 4))
print("RMSE on test set:", round(mean_squared_error(y_test, y_pred) ** 0.5, 4))


# Observation: The Linear Regression model was trained on the 80% training split. The printed coefficients table shows how much each feature contributes to the predicted yield (holding other features constant), and the feature with the highest positive coefficient is identified above. The model's R² and RMSE on the held-out test set are also reported to show how well it generalizes.
