# M5 Demand Forecasting - Local Setup Guide
# Author: Bekali Aslonov | bekali.aslonov@gmail.com

## Quick Start

### Step 1: Set Up Your Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Add M5 Data

1. Download data from [Kaggle M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
2. Create a `data` folder in root folder
3. Place these 3 files in the `data` folder:
   - calendar.csv
   - sell_prices.csv
   - sales_train_validation.csv

### Step 3: Run the Solution

```bash
python run_solution.py
```

## References

1. [M5 Competition Page](https://www.kaggle.com/c/m5-forecasting-accuracy)
2. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
3. [Tweedie Distribution Explained](https://en.wikipedia.org/wiki/Tweedie_distribution)