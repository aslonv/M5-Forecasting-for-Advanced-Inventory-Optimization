# M5 Demand Forecasting with Inventory Optimization

## Overview

In this repository I implement an ensemble forecasting system for the M5 competition dataset, combining recursive and direct forecasting strategies with LightGBM. The solution integrates demand forecasting with inventory optimization using an (s,S) policy framework.

**Author**: Bekali Aslonov | bekali.aslonov@gmail.com | [linkedin.com/in/aslonv](https://linkedin.com/in/aslonv)

## Technical Approach

### Forecasting Architecture

The system implements two complementary forecasting strategies:

1. **Recursive Model**: Single LightGBM model applied iteratively for multi-step ahead forecasting
2. **Direct Model**: Separate LightGBM model trained for each forecast horizon (h=1...14)

The final predictions use an ensemble average of both approaches, selected through time-series cross-validation.

### Model Specifications

**LightGBM Configuration**:

```python
{
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_data_in_leaf': 20
}
```

The Tweedie objective was selected to handle zero-inflated demand distributions (14% zero sales in the dataset).

### Feature Engineering

46 features across four categories:

**Temporal Features**:
- Lag features: {1, 2, 7, 14, 28} days
- Rolling statistics: mean and std over {7, 14, 28} day windows
- Calendar: dayofweek, dayofmonth, weekofyear, month, quarter, dayofyear

**Price Features**:
- 7-day price momentum: pct_change(7)
- Normalized price: (price - μ) / σ per item
- Raw sell price

**Categorical Encodings**:
- Label encoding for: item_id, dept_id, cat_id, store_id
- Binary indicators: is_weekend, is_holiday, snap_{CA,TX,WI}

**Hierarchical Aggregations**:
- store_item_avg: historical mean by (store, item)
- store_dept_avg: historical mean by (store, department)

## Performance Metrics

### Cross-Validation Results

| Model | CV1 RMSE | CV2 RMSE | CV3 RMSE | Mean | Std |
|-------|----------|----------|----------|------|-----|
| Recursive | 24.626 | 20.834 | 16.781 | 20.747 | 3.204 |
| Direct | 21.006 | 19.548 | 16.355 | 18.970 | 1.942 |
| Ensemble | 16.987 | 19.189 | 13.282 | 16.486 | 2.437 |

The ensemble approach achieved a 20.5% RMSE reduction compared to the recursive baseline.

### Inventory Optimization Results

Implemented (s,S) inventory policy with safety stock calculation:

```
s = μ_L + z × √(L × (σ²_demand + σ²_forecast))
S = s + 7 × μ_daily
```

**Parameters**:
- Lead time (L): 2 days
- Service level target: 95% (z = 1.645)
- Holding cost: $0.10/unit/day
- Stockout cost: $5.00/unit
- Ordering cost: $10.00/order

**Results**:
- Achieved service level: 98.6%
- Average inventory: 147 units
- Total cost per period: $231.09

## Implementation Details

### Cross-Validation Strategy

Time-series aware forward-chaining validation with 28-day windows:

```
Split 1: Train [1:800]   → Validate [801:828]
Split 2: Train [1:828]   → Validate [829:856]
Split 3: Train [1:856]   → Validate [857:884]
Test:    Train [1:884]   → Test [885:898]
```

### Computational Considerations

- Early stopping implemented with patience=50 (recursive) and patience=30 (direct)
- Minimum 28 observations required per series for lag feature validity
- Parallel training disabled for reproducibility (can be enabled via num_threads)

## Repository Structure

```
.
├── data/                           # M5 competition data (not included)
│   ├── calendar.csv
│   ├── sell_prices.csv
│   └── sales_train_validation.csv
├── outputs/                        # Generated artifacts
│   ├── m5_eda_analysis.png
│   ├── m5_forecasts.png
│   └── inventory_simulation.png
├── m5_forecasting_lgbm.py         # Main implementation
├── run_solution.py                # Entry point
├── requirements.txt               # Dependencies
└── README.md
```

## Installation and Usage

### Requirements

- Python 3.8+
- Dependencies: `lightgbm==4.0.0`, `pandas==2.0.3`, `numpy==1.24.3`, `scikit-learn==1.3.0`, `matplotlib==3.7.2`, `seaborn==0.12.2`, `scipy==1.11.1`

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/m5-forecasting.git
cd m5-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download M5 data from Kaggle
# Place calendar.csv, sell_prices.csv, sales_train_validation.csv in data/

# Run pipeline
python run_solution.py
```

### Output

The pipeline generates:

1. **EDA visualizations**: sales distributions, seasonality patterns, autocorrelations
2. **Forecast plots**: 14-day ahead predictions with confidence intervals
3. **Inventory simulation**: daily inventory levels with reorder points
4. **Performance metrics**: RMSE by model type and forecast horizon

## Key Findings

1. **Ensemble Superiority**: The ensemble consistently outperformed individual models across all CV splits, with lower variance in performance.

2. **Forecast Degradation**: Observed 18% RMSE increase from day 1 to day 14 forecasts, suggesting potential for horizon-specific feature engineering.

3. **Zero-Inflation Handling**: Tweedie loss with power=1.1 effectively handled the 14% zero-sales observations without requiring separate classification.

4. **Feature Importance**: Lag-1 and lag-7 features dominated importance scores, confirming strong daily and weekly patterns in retail demand.

## Limitations and Future Work

### Current Limitations

- Single-store evaluation (CA_1 → CA_2)
- Top 5 items only for computational efficiency
- No cross-item or cannibalization effects modeled
- Static hyperparameters across all items

## References

1. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
2. Seeger, M. W., et al. (2016). Bayesian Intermittent Demand Forecasting for Large Inventories. *NeurIPS*.
3. Januschowski, T., et al. (2020). Criteria for Classifying Forecasting Methods. *International Journal of Forecasting*.
4. M5 Competition: https://www.kaggle.com/c/m5-forecasting-accuracy
