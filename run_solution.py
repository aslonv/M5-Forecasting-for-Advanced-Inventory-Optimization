#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from m5_forecasting_lgbm import main

if __name__ == "__main__":
    print("\nStarting M5 Forecasting Solution...")
    
    if not os.path.exists('data'):
        print("\n  ERROR: 'data' folder not found!")
        print("Please create a 'data' folder and add the M5 CSV files:")
        print("  - calendar.csv")
        print("  - sell_prices.csv")
        print("  - sales_train_validation.csv")
        sys.exit(1)
    
    required_files = ['calendar.csv', 'sell_prices.csv', 'sales_train_validation.csv']
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join('data', file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n  ERROR: Missing data files in 'data' folder:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download these files from Kaggle M5 competition")
        sys.exit(1)
    
    try:
        pipeline = main()
        print("Check the outputs/ folder for results:")
        print("- m5_eda_analysis.png")
        print("- m5_forecasts.png") 
        print("- inventory_simulation.png")
        print("\n")
    except Exception as e:
        print(f"\n  Error during execution: {str(e)}")
        raise e