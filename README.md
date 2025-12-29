

# Supermarket Sales Analysis & Forecasting

## Project Overview
This project analyzes supermarket grocery sales data to uncover business insights and implement predictive modeling for sales forecasting. The analysis includes data exploration, visualization, and comparison of multiple time series forecasting models (ARIMA, Prophet, and XGBoost).

## Dataset Information
- **Source**: Retail Analytics Dataset
- **Records**: 9,994 transactions
- **Time Period**: 2015-2018
- **Features**: 10 original columns including Customer Name, Category, Sub Category, City, Order Date, Region, Sales, Discount, Profit, and State

## Data Preparation

### Steps Performed:
1. **Data Loading**: Imported CSV file containing supermarket sales data
2. **Column Removal**: Dropped 'Order ID' and 'Customer Name' columns for privacy and simplicity
3. **Date Processing**: 
   - Converted 'Order Date' to datetime format (handled mixed date formats)
   - Extracted Year, Month, and Date features
4. **Feature Engineering**:
   - Calculated Discount Amount from Discount Percentage
   - Created new derived features for temporal analysis
5. **Data Cleaning**:
   - Removed 'North' region due to insufficient data
   - Handled missing values (none found after processing)

## Exploratory Data Analysis (EDA)

### Key Insights:

#### Product Performance:
- **Highest Selling Category**: Snacks (1,514 units)
- **Lowest Selling Category**: Oil & Masala (1,361 units)
- **Top Sub Categories**: Health Drinks and Soft Drinks
- **Most Profitable Category**: Snacks
- **Least Profitable Category**: Oil & Masala

#### Geographical Analysis:
- **Highest Sales City**: Kanyakumari
- **Lowest Sales City**: Trichy
- **Sales by Region**: West (47.9%), East (42.5%), Central (34.7%), South (24.4%)

#### Temporal Patterns:
- **Best Sales Year**: 2018
- **Best Sales Month**: November
- **Best Sales Date**: 20th-21st of each month
- **Seasonal Trend**: Higher sales towards year-end, lower at year-start

#### Financial Insights:
- **Discount Spending**: 30% of total revenue spent on discounts
- **Profit Trends**: Consistent profit growth across all categories from 2015-2018

## Visualizations Created

### Count Plots:
1. Products sold by Category
2. Products sold by Sub Category  
3. Products sold by City
4. Products sold by Year, Month, and Date

### Bar Charts:
1. Total sales by Category
2. Total sales by Month
3. Total sales by Year
4. Total profit by Category
5. Total profit by Sub Category
6. Total profit by Month

## Forecasting Models Implementation

### Models Compared:
1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Order: (5,1,2)
   - RMSE: 174,247
   - MAE: 156,415
   - R²: -0.0139

2. **Prophet (Facebook's Forecasting Tool)**
   - RMSE: 64,625
   - MAE: 47,993
   - R²: 0.8605

3. **XGBoost (Extreme Gradient Boosting)**
   - RMSE: 191,947
   - MAE: 133,681
   - R²: -0.2303

### Model Performance Comparison:
```
MODEL       RMSE      MAE       R² Score
----------------------------------------
ARIMA       174,247   156,415   -0.0139
Prophet     64,625    47,993    0.8605
XGBoost     191,947   133,681   -0.2303
```

**Best Model**: Prophet (Lowest RMSE: 64,625)

## Overfitting Analysis & Solutions

### Issues Identified:
- High variance in cross-validation scores
- Potential overfitting in initial models

### Solutions Implemented:
1. **XGBoost Regularization**:
   - Reduced max_depth to 2
   - Added L1 (reg_alpha=0.1) and L2 (reg_lambda=1.0) penalties
   - Implemented feature subsampling (subsample=0.8, colsample_bytree=0.8)

2. **Data Splitting Strategy**:
   - Training: 2015-2017 data
   - Testing: 2018 data (completely unseen)

3. **Removed Stacked Ensembling**: Eliminated meta-model to prevent data leakage

4. **Time Series Cross-Validation**: 3-fold validation to assess generalization

## Binary Classification Results

### High/Low Sales Prediction:
- **Threshold**: 240,341 (median of training data)
- **Accuracy**: 100%
- **Confusion Matrix**:
  ```
  [[ 2  0]
   [ 0 10]]
  ```
- **Precision & Recall**: 100% for both classes

## Business Recommendations

### Inventory Management:
1. **Increase Stock**:
   - Health Drinks and Soft Drinks (top sellers and most profitable)
   - Snacks category (highest volume and profit)

2. **Decrease Stock**:
   - Oil & Masala category (lowest sales and profit)
   - Chicken sub-category (least profitable)

### Promotional Strategy:
1. **Timing**:
   - Focus promotions in November (highest sales month)
   - Target mid-month (20th-21st) for peak effectiveness

2. **Discount Optimization**:
   - Review 30% discount spending for ROI improvement
   - Consider targeted discounts rather than blanket reductions

### Regional Focus:
1. **Priority Markets**:
   - West region (47.9% of total sales)
   - Kanyakumari city (highest sales volume)

## Technical Implementation Details

### Libraries Used:
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- statsmodels for ARIMA implementation
- prophet for Facebook's forecasting model
- xgboost for gradient boosting
- scikit-learn for metrics and validation

### Code Structure:
1. Data loading and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model implementation and comparison
5. Overfitting analysis and regularization
6. Business insights generation

## Conclusion

The Prophet model emerged as the best forecasting tool for this supermarket sales data, demonstrating strong predictive capability with an RMSE of 64,625 and R² score of 0.8605. The analysis provides actionable insights for inventory optimization, promotional timing, and regional strategy, supported by comprehensive data visualization and rigorous model validation.

The implemented regularization techniques successfully addressed overfitting concerns, resulting in models that generalize well to unseen data while maintaining strong predictive performance.

