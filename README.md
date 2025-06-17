# Car Price Prediction Project

## Overview
This project aims to predict the selling price of used cars based on their specifications. It involves data exploration, feature engineering, and the training of multiple regression models to evaluate their performance.

## Repository Structure
```
carpricepredictor/
├── CarPrice_Assignment.csv      # Main dataset
├── Data Dictionary - carprices.xlsx  # Data dictionary
├── car_price_prediction.py      # Main analysis script
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── Visuals/                     # Generated visualizations
    ├── correlation_matrix.png
    ├── price_distribution.png
    ├── price_vs_horsepower.png
    ├── price_by_carbody.png
    ├── price_vs_power_weight.png
    ├── price_vs_mpg.png
    ├── feature_importance.png
    ├── residual_plot.png
    ├── actual_vs_predicted.png
    └── model_comparison.png
```

## Data Description
- **Dataset:** `CarPrice_Assignment.csv`
- **Data Dictionary:** `Data Dictionary - carprices.xlsx`
- **Shape:** 205 rows, 26 columns
- **Features:** Includes both categorical (e.g., CarName, fueltype, carbody) and numerical (e.g., horsepower, wheelbase, price) features.
- **Missing Values:** None

## Features
- **Categorical Features:** CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem
- **Numerical Features:** symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg, price
- **Engineered Features:** 
  - power_to_weight (horsepower/curbweight)
  - mpg_avg (average of city and highway MPG)

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd carpricepredictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the prediction script:
```bash
python car_price_prediction.py
```

## Visualizations
The script generates several visualizations saved in the `Visuals` directory:

1. **Correlation Matrix** (`correlation_matrix.png`)
   - Shows relationships between numerical features
   - Helps identify multicollinearity

2. **Price Distribution** (`price_distribution.png`)
   - Distribution of car prices
   - Includes KDE plot for better visualization

3. **Price vs Horsepower** (`price_vs_horsepower.png`)
   - Scatter plot colored by fuel type
   - Shows relationship between power and price

4. **Price by Car Body Type** (`price_by_carbody.png`)
   - Box plot showing price distribution for different car body types
   - Helps identify price patterns by car category

5. **Price vs Power-to-Weight Ratio** (`price_vs_power_weight.png`)
   - Scatter plot colored by engine type
   - Shows relationship between performance and price

6. **Price vs Average MPG** (`price_vs_mpg.png`)
   - Scatter plot colored by drive wheel type
   - Shows relationship between fuel efficiency and price

7. **Feature Importance** (`feature_importance.png`)
   - Bar plot of top 15 most important features
   - Based on model coefficients

8. **Residual Plot** (`residual_plot.png`)
   - Shows model prediction errors
   - Helps assess model fit

9. **Actual vs Predicted Prices** (`actual_vs_predicted.png`)
   - Scatter plot comparing actual and predicted prices
   - Includes perfect prediction line

10. **Model Comparison** (`model_comparison.png`)
    - Bar plots comparing RMSE and R² scores
    - Shows performance across different models

## Model Performance
- **Linear Regression:**  
  - RMSE: $6,211.60  
  - R²: 0.5112  
  - MAE: $3,821.80

- **Ridge Regression:**  
  - RMSE: $3,156.98  
  - R²: 0.8738

- **Lasso Regression:**  
  - RMSE: $4,126.06  
  - R²: 0.7843

## Future Improvements
- Experiment with additional feature engineering
- Try other regression models (e.g., Random Forest, Gradient Boosting)
- Hyperparameter tuning for Ridge and Lasso regression
- Cross-validation to ensure robust model performance
- Add interactive visualizations using Plotly
- Implement model persistence for future predictions


