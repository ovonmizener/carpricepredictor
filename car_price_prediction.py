import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading data...")
df = pd.read_csv('CarPrice_Assignment.csv')

# Display basic information about the dataset
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\nCategorical columns:", list(categorical_cols))
print("\nNumerical columns:", list(numerical_cols))

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

# Remove car_ID as it's not useful for prediction
df_processed = df_processed.drop('car_ID', axis=1)

# Create new features
df_processed['power_to_weight'] = df_processed['horsepower'] / df_processed['curbweight']
df_processed['mpg_avg'] = (df_processed['citympg'] + df_processed['highwaympg']) / 2

# Create Visualizations
print("\nGenerating visualizations...")

# 1. Correlation Matrix
plt.figure(figsize=(15, 10))
correlation_matrix = df_processed.select_dtypes(include=['float64', 'int64']).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Correlation Matrix of Numerical Features', pad=20)
plt.tight_layout()
plt.savefig('Visuals/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df_processed, x='price', bins=30, kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.savefig('Visuals/price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Price vs Horsepower
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_processed, x='horsepower', y='price', hue='fueltype', alpha=0.6)
plt.title('Price vs Horsepower by Fuel Type')
plt.xlabel('Horsepower')
plt.ylabel('Price ($)')
plt.savefig('Visuals/price_vs_horsepower.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Box Plot of Prices by Car Body Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_processed, x='carbody', y='price')
plt.title('Price Distribution by Car Body Type')
plt.xlabel('Car Body Type')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.savefig('Visuals/price_by_carbody.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Power to Weight Ratio vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_processed, x='power_to_weight', y='price', hue='enginetype', alpha=0.6)
plt.title('Price vs Power-to-Weight Ratio by Engine Type')
plt.xlabel('Power-to-Weight Ratio')
plt.ylabel('Price ($)')
plt.savefig('Visuals/price_vs_power_weight.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Average MPG vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_processed, x='mpg_avg', y='price', hue='drivewheel', alpha=0.6)
plt.title('Price vs Average MPG by Drive Wheel Type')
plt.xlabel('Average MPG')
plt.ylabel('Price ($)')
plt.savefig('Visuals/price_vs_mpg.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Feature Importance Plot
plt.figure(figsize=(12, 6))
feature_names = (model.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(X.select_dtypes(include=['object']).columns))
feature_names = np.concatenate([X.select_dtypes(include=['float64', 'int64']).columns, feature_names])

coefficients = model.named_steps['regressor'].coef_

# 8. Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('Visuals/residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.savefig('Visuals/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# Try Ridge and Lasso regression
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=1.0))
])

# Train and evaluate Ridge
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

# Train and evaluate Lasso
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)

print("\nComparison of Models:")
print(f"Linear Regression - RMSE: ${rmse:,.2f}, R²: {r2:.4f}")
print(f"Ridge Regression - RMSE: ${ridge_rmse:,.2f}, R²: {ridge_r2:.4f}")
print(f"Lasso Regression - RMSE: ${lasso_rmse:,.2f}, R²: {lasso_r2:.4f}")

# 10. Model Comparison Plot
plt.figure(figsize=(10, 6))
models = ['Linear', 'Ridge', 'Lasso']
rmse_values = [rmse, ridge_rmse, lasso_rmse]
r2_values = [r2, ridge_r2, lasso_r2]

x = np.arange(len(models))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax1.bar(x - width/2, rmse_values, width, label='RMSE')
ax1.set_ylabel('RMSE ($)')
ax1.set_title('Model Comparison - RMSE')
ax1.set_xticks(x)
ax1.set_xticklabels(models)

ax2.bar(x - width/2, r2_values, width, label='R²')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Comparison - R² Score')
ax2.set_xticks(x)
ax2.set_xticklabels(models)

plt.tight_layout()
plt.savefig('Visuals/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Prepare features for modeling
X = df_processed.drop(['price'], axis=1)
y = df_processed['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['float64', 'int64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
    ])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: ${mae:,.2f}")

# Get feature importances
feature_names = (model.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(X.select_dtypes(include=['object']).columns))
feature_names = np.concatenate([X.select_dtypes(include=['float64', 'int64']).columns, feature_names])

coefficients = model.named_steps['regressor'].coef_

# Create a DataFrame of feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Additional visualizations
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('residual_plot.png')
plt.close()

# Try Ridge and Lasso regression
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=1.0))
])

# Train and evaluate Ridge
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

# Train and evaluate Lasso
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)

print("\nComparison of Models:")
print(f"Linear Regression - RMSE: ${rmse:,.2f}, R²: {r2:.4f}")
print(f"Ridge Regression - RMSE: ${ridge_rmse:,.2f}, R²: {ridge_r2:.4f}")
print(f"Lasso Regression - RMSE: ${lasso_rmse:,.2f}, R²: {lasso_r2:.4f}")

# Additional visualizations
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.savefig('actual_vs_predicted.png')
plt.close() 