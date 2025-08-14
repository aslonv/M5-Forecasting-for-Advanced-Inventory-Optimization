#!/usr/bin/env python3

import os
os.environ['TCL_LIBRARY'] = r'C:\Program Files\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Program Files\Python313\tcl\tk8.6'
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

class M5ForecastingPipeline:
    """
    Pipeline:
    - LightGBM with Tweedie loss
    - Recursive and non-recursive approaches
    - Robust CV strategy
    - Hierarchical modeling capability
    """
    
    def __init__(self, store_id='CA_1', n_items=5, test_store_id='CA_2'):
        self.store_id = store_id
        self.test_store_id = test_store_id
        self.n_items = n_items
        self.data = {}
        self.models = {}
        self.predictions = {}
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        
    def load_and_preprocess_data(self):
        print("Loading M5 dataset...")
        print(f"   Looking for data in: {self.data_dir}")
        
        try:
            self.data['calendar'] = pd.read_csv(os.path.join(self.data_dir, 'calendar.csv'))
            self.data['prices'] = pd.read_csv(os.path.join(self.data_dir, 'sell_prices.csv'))
            self.data['sales_train'] = pd.read_csv(os.path.join(self.data_dir, 'sales_train_validation.csv'))
            
            print(f"   Data loaded successfully!")
            print(f"   Calendar: {self.data['calendar'].shape}")
            print(f"   Prices: {self.data['prices'].shape}")
            print(f"   Sales: {self.data['sales_train'].shape}")
        except FileNotFoundError as e:
            print(f"   Error: Could not find data files in {self.data_dir}")
            print(f"   Please ensure calendar.csv, sell_prices.csv, and sales_train_validation.csv are in the data/ folder")
            raise e
        
        self._select_top_items()
        
        self._prepare_long_format()
        
        self._engineer_features()
        
        return self.data['processed']
    
    def _select_top_items(self):
        """Select top N selling items from the specified store"""
        store_data = self.data['sales_train'][
            self.data['sales_train']['store_id'] == self.store_id
        ]
        
        # Calculate total sales per item
        sales_cols = [col for col in store_data.columns if col.startswith('d_')]
        item_sales = store_data[['item_id'] + sales_cols].set_index('item_id')
        total_sales = item_sales.sum(axis=1).sort_values(ascending=False)
        
        # Select top N items
        self.top_items = total_sales.head(self.n_items).index.tolist()
        print(f"\nSelected top {self.n_items} items from {self.store_id}:")
        for i, item in enumerate(self.top_items, 1):
            print(f"   {i}. {item}: {total_sales[item]:,.0f} units")
        
        # Also get data for test store (same items)
        self.data['train_store'] = store_data[store_data['item_id'].isin(self.top_items)]
        test_store_data = self.data['sales_train'][
            (self.data['sales_train']['store_id'] == self.test_store_id) &
            (self.data['sales_train']['item_id'].isin(self.top_items))
        ]
        self.data['test_store'] = test_store_data
    
    def _prepare_long_format(self):
        print("\nPreparing data in long format...")
        
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        sales_cols = [col for col in self.data['train_store'].columns if col.startswith('d_')]
        
        # Process both train and test stores
        all_data = []
        for store_name, store_data in [('train', self.data['train_store']), 
                                       ('test', self.data['test_store'])]:
            melted = pd.melt(
                store_data,
                id_vars=id_cols,
                value_vars=sales_cols,
                var_name='d',
                value_name='sales'
            )
            melted['data_type'] = store_name
            all_data.append(melted)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        combined_data = combined_data.merge(self.data['calendar'], on='d', how='left')
        
        combined_data = combined_data.merge(
            self.data['prices'],
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )

        # Handle missing prices
        combined_data['sell_price'] = combined_data.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        
        combined_data = combined_data.sort_values(['store_id', 'item_id', 'date'])
        
        self.data['processed'] = combined_data
        print(f"   Processed data shape: {combined_data.shape}")
    
    def _engineer_features(self):
        print("\nEngineering features...")
        df = self.data['processed']
        # Calendar features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['dayofyear'] = df['date'].dt.dayofyear
        
        # Price features
        df['price_momentum'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: x.pct_change(periods=7).fillna(0)
        )
        df['price_norm'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-7)
        )
        
        # Weekend and holiday flags
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_holiday'] = (~df['event_name_1'].isna()).astype(int)
        
        # SNAP features
        df['snap_CA'] = df['snap_CA'].fillna(0).astype(int)
        df['snap_TX'] = df['snap_TX'].fillna(0).astype(int)
        df['snap_WI'] = df['snap_WI'].fillna(0).astype(int)
        
        # Lag features (both recursive and non-recursive will use these)
        for lag in [1, 2, 7, 14, 28]:
            df[f'lag_{lag}'] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag)
        
        for window in [7, 14, 28]:
            df[f'rolling_mean_{window}'] = df.groupby(['store_id', 'item_id'])['sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['store_id', 'item_id'])['sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )
        
        # Store-level features
        df['store_item_avg'] = df.groupby(['store_id', 'item_id'])['sales'].transform('mean')
        df['store_dept_avg'] = df.groupby(['store_id', 'dept_id'])['sales'].transform('mean')
        
        for cat in ['item_id', 'dept_id', 'cat_id', 'store_id']:
            df[f'{cat}_encoded'] = pd.Categorical(df[cat]).codes
        
        min_lag = 28  
        self.data['processed'] = df[df.groupby(['store_id', 'item_id']).cumcount() >= min_lag].copy()
        
        print(f"   Final processed data shape: {self.data['processed'].shape}")
        print(f"   Features created: {len([col for col in df.columns if col not in ['date', 'sales', 'd']])} features")
    
    def exploratory_analysis(self):
        print("\nPerforming exploratory data analysis...")
        
        train_data = self.data['processed'][self.data['processed']['data_type'] == 'train']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'M5 Exploratory Data Analysis - Store {self.store_id}', fontsize=16)
        
        # Sales distribution (log scale to handle zeros)
        ax = axes[0, 0]
        sales_nonzero = train_data[train_data['sales'] > 0]['sales']
        sales_nonzero.hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Daily Sales (non-zero)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Sales Distribution (Zero sales: {(train_data["sales"]==0).mean():.1%})')
        ax.set_yscale('log')
        
        # Time series for each item
        ax = axes[0, 1]
        for item in self.top_items[:3]:  # Show first 3 items
            item_data = train_data[train_data['item_id'] == item].groupby('date')['sales'].sum()
            ax.plot(item_data.index, item_data.values, label=item, alpha=0.8)
        ax.set_title('Daily Sales by Item')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        
        # Weekly seasonality
        ax = axes[1, 0]
        weekly_sales = train_data.groupby('dayofweek')['sales'].agg(['mean', 'std'])
        weekly_sales['mean'].plot(kind='bar', ax=ax, color='darkblue', yerr=weekly_sales['std'], capsize=5)
        ax.set_title('Sales by Day of Week')
        ax.set_xlabel('Day of Week (0=Monday)')
        ax.set_ylabel('Average Sales')
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        
        # Price elasticity
        ax = axes[1, 1]
        price_sales = train_data.groupby(['item_id', 'sell_price'])['sales'].mean().reset_index()
        for item in self.top_items[:3]:
            item_data = price_sales[price_sales['item_id'] == item]
            ax.scatter(item_data['sell_price'], item_data['sales'], label=item, alpha=0.6, s=50)
        ax.set_title('Price vs Sales Relationship')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Average Sales')
        ax.legend()
        
        # Holiday impact
        ax = axes[2, 0]
        holiday_impact = train_data.groupby('is_holiday')['sales'].mean()
        holiday_impact.plot(kind='bar', ax=ax, color=['gray', 'red'])
        ax.set_title('Holiday Impact on Sales')
        ax.set_xlabel('Is Holiday')
        ax.set_ylabel('Average Sales')
        ax.set_xticklabels(['Regular Day', 'Holiday'], rotation=0)
        
        # Autocorrelation (for one item)
        ax = axes[2, 1]
        item_sales = train_data[train_data['item_id'] == self.top_items[0]].set_index('date')['sales']
        autocorr = [item_sales.autocorr(lag=i) for i in range(1, 29)]
        ax.bar(range(1, 29), autocorr, color='darkgreen', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f'Autocorrelation for {self.top_items[0]}')
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Autocorrelation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'm5_eda_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   EDA visualization saved to outputs/m5_eda_analysis.png")
        
        print("\nKey Statistics:")
        print(f"   - Zero sales percentage: {(train_data['sales'] == 0).mean():.1%}")
        print(f"   - Average daily sales: {train_data['sales'].mean():.2f}")
        print(f"   - Sales standard deviation: {train_data['sales'].std():.2f}")
        print(f"   - Weekend sales lift: {(holiday_impact[1] / holiday_impact[0] - 1):.1%}")
    
    def create_cv_splits(self):
        print("\nCreating time-based CV splits...")
        
        dates = self.data['processed']['date'].unique()
        dates = np.sort(dates)
        
        n_days = len(dates)
        cv_days = 28
        
        splits = []
        
        # Create 3 CV splits + 1 test split
        for i in range(4):
            val_end = n_days - i * cv_days
            val_start = val_end - cv_days
            train_end = val_start
            
            if train_end < 200:  # Ensures enough training data
                break
                
            splits.append({
                'name': f'cv{4-i}' if i < 3 else 'test',
                'train_dates': dates[:train_end],
                'val_dates': dates[val_start:val_end],
                'train_end': dates[train_end-1],
                'val_start': dates[val_start],
                'val_end': dates[val_end-1]
            })
        
        splits.reverse()  
        
        print("   CV Splits created:")
        for split in splits:
            train_end = pd.Timestamp(split['train_end'])
            val_start = pd.Timestamp(split['val_start'])
            val_end = pd.Timestamp(split['val_end'])
            print(f"     {split['name']}: train until {train_end.strftime('%Y-%m-%d')}, "
                f"validate {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        
        self.cv_splits = splits
        return splits
    
    def train_recursive_model(self):
        """Train recursive LightGBM model (single model used iteratively)"""
        print("\nTraining recursive model...")
        
        features = [
            # Calendar
            'dayofweek', 'dayofmonth', 'weekofyear', 'month', 'quarter', 'dayofyear',
            # Price
            'sell_price', 'price_momentum', 'price_norm',
            # Flags
            'is_weekend', 'is_holiday', 'snap_CA',
            # Lags
            'lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_28',
            # Rolling
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
            # Hierarchical
            'store_item_avg', 'store_dept_avg',
            # Encoded categoricals
            'item_id_encoded', 'dept_id_encoded', 'cat_id_encoded', 'store_id_encoded'
        ]
        
        train_data = self.data['processed'][
            (self.data['processed']['data_type'] == 'train') &
            (self.data['processed']['date'] <= self.cv_splits[-1]['train_end'])
        ]
        
        X_train = train_data[features]
        y_train = train_data['sales']
        
        # LightGBM parameters optimized for Tweedie
        params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,  
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 20
        }
        
        train_set = lgb.Dataset(X_train, label=y_train)
        self.models['recursive'] = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[train_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        print(f"   Recursive model trained with {self.models['recursive'].num_trees()} trees")
        
        self.features = features
        
    def train_direct_models(self, horizon=14):
        """Train direct (non-recursive) models for each forecast horizon"""
        print(f"\nTraining direct models for {horizon} day horizon...")
        
        self.models['direct'] = {}
        
        train_data = self.data['processed'][
            (self.data['processed']['data_type'] == 'train') &
            (self.data['processed']['date'] <= self.cv_splits[-1]['train_end'])
        ].copy()
        
        # Creates a validation split (use last 10% of training data)
        split_date = train_data['date'].quantile(0.9)
        train_split = train_data[train_data['date'] < split_date]
        val_split = train_data[train_data['date'] >= split_date]
        
        for h in range(1, horizon + 1):
            print(f"   Training model for day {h}...", end='')
            
            # Creates future target for both train and validation
            train_split[f'target_d{h}'] = train_split.groupby(['store_id', 'item_id'])['sales'].shift(-h)
            val_split[f'target_d{h}'] = val_split.groupby(['store_id', 'item_id'])['sales'].shift(-h)
            
            train_h = train_split.dropna(subset=[f'target_d{h}'])
            val_h = val_split.dropna(subset=[f'target_d{h}'])
            
            X_train = train_h[self.features]
            y_train = train_h[f'target_d{h}']
            X_val = val_h[self.features]
            y_val = val_h[f'target_d{h}']
            
            params = {
                'objective': 'tweedie',
                'tweedie_variance_power': 1.1,
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbose': -1,
                'num_threads': -1
            }
            
            train_set = lgb.Dataset(X_train, label=y_train)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            
            model = lgb.train(
                params,
                train_set,
                num_boost_round=300,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            
            self.models['direct'][h] = model
            print(f" {model.num_trees()} trees")

    def evaluate_models(self, approach='ensemble'):
        print(f"\nEvaluating models with {approach} approach...")
        
        cv_scores = {'recursive': [], 'direct': [], 'ensemble': []}
        
        for split in self.cv_splits[:-1]:  
            print(f"\nEvaluating on {split['name']}...")
            
            # Gets validation data
            val_data = self.data['processed'][
                (self.data['processed']['data_type'] == 'train') &
                (self.data['processed']['date'].isin(split['val_dates']))
            ].copy()
            
            print(f"  Validation data shape: {val_data.shape}")
            print(f"  Validation date range: {val_data['date'].min()} to {val_data['date'].max()}")
            
            last_val_date = val_data['date'].max()
            future_start = last_val_date + timedelta(days=1)
            future_end = last_val_date + timedelta(days=14)
            
            future_data = self.data['processed'][
                (self.data['processed']['data_type'] == 'train') &
                (self.data['processed']['date'] >= future_start) &
                (self.data['processed']['date'] <= future_end)
            ]
            
            print(f"  Future data available: {len(future_data)} rows")
            print(f"  Future date range needed: {future_start} to {future_end}")
            
            if len(future_data) == 0:
                print(f"  WARNING: No future data available for evaluation!")
                # Use penalty score
                for approach_name in ['recursive', 'direct', 'ensemble']:
                    cv_scores[approach_name].append(999.0)
                continue
 
            val_first_day = val_data.groupby(['store_id', 'item_id']).first().reset_index()

            rec_preds = self._predict_recursive(val_first_day, horizon=14)
             
            dir_preds = self._predict_direct(val_first_day, horizon=14)
            
            ens_preds = (rec_preds + dir_preds) / 2
            
            for approach_name, preds in [('recursive', rec_preds), 
                                        ('direct', dir_preds), 
                                        ('ensemble', ens_preds)]:
                errors = []
                
                # For each item, get predictions and actuals
                for idx, row in val_first_day.iterrows():
                    store_id = row['store_id']
                    item_id = row['item_id']
                    val_date = row['date']
                    
                    # Gets 14-day forecast for this item
                    if idx in preds.index:
                        item_preds = preds.loc[idx].values
                        
                        # Gets actual values for next 14 days
                        future_dates = pd.date_range(val_date + timedelta(days=1), periods=14, freq='D')
                        actuals = []
                        
                        for future_date in future_dates:
                            actual_data = self.data['processed'][
                                (self.data['processed']['store_id'] == store_id) &
                                (self.data['processed']['item_id'] == item_id) &
                                (self.data['processed']['date'] == future_date)
                            ]
                            
                            if len(actual_data) > 0:
                                actuals.append(actual_data['sales'].iloc[0])
                            else:
                                actuals.append(np.nan)
                        
                        actuals = np.array(actuals)
                        if not np.any(np.isnan(actuals)):
                            item_errors = (item_preds - actuals) ** 2
                            errors.extend(item_errors)
                
                if errors:
                    rmse = np.sqrt(np.mean(errors))
                    cv_scores[approach_name].append(rmse)
                    print(f"    {approach_name}: {len(errors)//14} items evaluated, RMSE={rmse:.3f}")
                else:
                    cv_scores[approach_name].append(999.0)
                    print(f"    {approach_name}: No valid items for evaluation")
                    
            print(f"  RMSE - Recursive: {cv_scores['recursive'][-1]:.3f}, "
                f"Direct: {cv_scores['direct'][-1]:.3f}, "
                f"Ensemble: {cv_scores['ensemble'][-1]:.3f}")
        
        print("\nCV Summary (lower is better):")
        for approach_name in ['recursive', 'direct', 'ensemble']:
            scores = [s for s in cv_scores[approach_name] if s < 999]  
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores) if len(scores) > 1 else 0
                print(f"  {approach_name:10s}: mean={mean_score:.3f}, std={std_score:.3f}, "
                    f"mean+std={mean_score + std_score:.3f} ({len(scores)} valid splits)")
            else:
                print(f"  {approach_name:10s}: No valid evaluation data")
   
        # Determine best approach - if no valid scores, default to ensemble
        valid_approaches = {k: v for k, v in cv_scores.items() if any(s < 999 for s in v)}
        if valid_approaches:
            best_approach = min(valid_approaches.keys(), 
                            key=lambda x: np.mean([s for s in cv_scores[x] if s < 999]))
        else:
            best_approach = 'ensemble'  
            
        print(f"\n  BEST APPROACH: {best_approach}")
        
        self.best_approach = best_approach
        self.cv_scores = cv_scores
        
        return cv_scores
    
    def _predict_recursive(self, data, horizon=14):
        predictions = []
        current_features = data[self.features].copy()
        
        for h in range(horizon):
            # Predict next day
            pred = self.models['recursive'].predict(current_features)
            predictions.append(pred)
            
            # Update features for next prediction
            if h < horizon - 1:
                current_features['lag_28'] = current_features['lag_14'].values
                current_features['lag_14'] = current_features['lag_7'].values
                current_features['lag_7'] = current_features['lag_2'].values
                current_features['lag_2'] = current_features['lag_1'].values
                current_features['lag_1'] = pred
                
                for window in [7, 14, 28]:
                    current_features[f'rolling_mean_{window}'] = (
                        current_features[f'rolling_mean_{window}'] * (window - 1) + pred
                    ) / window
        
        return pd.DataFrame(predictions).T
    
    def _predict_direct(self, data, horizon=14):
        predictions = []
        current_features = data[self.features].copy()
        
        for h in range(1, horizon + 1):
            if h in self.models['direct']:
                pred = self.models['direct'][h].predict(current_features)
                predictions.append(pred)
            else:
                # Fallback to last available model
                pred = self.models['direct'][max(self.models['direct'].keys())].predict(current_features)
                predictions.append(pred)
        
        return pd.DataFrame(predictions).T
    
    def make_final_predictions(self):
        print(f"\nMaking final predictions using {self.best_approach} approach...")
        
        test_data = self.data['processed'][
            self.data['processed']['data_type'] == 'test'
        ].copy()
        
        last_dates = test_data.groupby(['store_id', 'item_id'])['date'].max()
        
        predictions = []
        for (store, item), last_date in last_dates.items():
            item_data = test_data[
                (test_data['store_id'] == store) & 
                (test_data['item_id'] == item) &
                (test_data['date'] == last_date)
            ]
            
            if self.best_approach == 'recursive':
                pred = self._predict_recursive(item_data, horizon=14)
            elif self.best_approach == 'direct':
                pred = self._predict_direct(item_data, horizon=14)
            else:  
                rec = self._predict_recursive(item_data, horizon=14)
                dir = self._predict_direct(item_data, horizon=14)
                pred = (rec + dir) / 2
            
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=14, freq='D')
            item_forecast = pd.DataFrame({
                'store_id': store,
                'item_id': item,
                'date': forecast_dates,
                'forecast': pred.values.flatten()
            })
            predictions.append(item_forecast)
        
        self.predictions['final'] = pd.concat(predictions, ignore_index=True)
        
        print(f"  Generated {len(self.predictions['final'])} predictions")
        
        return self.predictions['final']
    
    def visualize_forecasts(self):
        print("\nCreating forecast visualizations...")
        
        fig, axes = plt.subplots(self.n_items, 1, figsize=(12, 4*self.n_items))
        if self.n_items == 1:
            axes = [axes]
        
        for idx, item in enumerate(self.top_items):
            ax = axes[idx]
            
            hist_data = self.data['processed'][
                (self.data['processed']['item_id'] == item) &
                (self.data['processed']['store_id'] == self.test_store_id) &
                (self.data['processed']['data_type'] == 'test')
            ].tail(60)
            
            forecast_data = self.predictions['final'][
                (self.predictions['final']['item_id'] == item) &
                (self.predictions['final']['store_id'] == self.test_store_id)
            ]
            
            ax.plot(hist_data['date'], hist_data['sales'], 'b-', label='Historical', linewidth=2)
            
            hist_std = hist_data['sales'].std()
            ax.plot(forecast_data['date'], forecast_data['forecast'], 'r--', 
                   label='Forecast', linewidth=2)
            ax.fill_between(
                forecast_data['date'],
                forecast_data['forecast'] - 1.96 * hist_std,
                forecast_data['forecast'] + 1.96 * hist_std,
                alpha=0.3, color='red', label='95% CI'
            )
            
            ax.set_title(f'Forecast for {item} at {self.test_store_id}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'm5_forecasts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Forecast visualization saved to outputs/m5_forecasts.png")
    
    def simulate_inventory_advanced(self, lead_time=2, holding_cost=0.1, stockout_cost=5.0, 
                                  ordering_cost=10.0, service_level_target=0.95):
        print(f"\nSimulating inventory with service level target: {service_level_target:.1%}")
        
        results = []
        
        for item in self.top_items:
            hist_data = self.data['processed'][
                (self.data['processed']['item_id'] == item) &
                (self.data['processed']['store_id'] == self.test_store_id) &
                (self.data['processed']['data_type'] == 'test')
            ].tail(90)
            
            forecast_data = self.predictions['final'][
                (self.predictions['final']['item_id'] == item) &
                (self.predictions['final']['store_id'] == self.test_store_id)
            ]
            
            # Calculate demand statistics
            mean_demand = hist_data['sales'].mean()
            std_demand = hist_data['sales'].std()
            
            forecast_mae = mean_demand * 0.3  # Approximate 30% MAPE
            
            from scipy.stats import norm
            z_score = norm.ppf(service_level_target)
            
            safety_stock = z_score * np.sqrt(
                lead_time * (std_demand**2 + forecast_mae**2)
            )
            
            # Set reorder point and order-up-to level
            s = mean_demand * lead_time + safety_stock
            S = s + mean_demand * 7 # One week of additional inventory
            
            sim_results = self._run_inventory_simulation(
                forecast_data['forecast'].values,
                s, S, lead_time,
                holding_cost, stockout_cost, ordering_cost
            )
            
            sim_results['item_id'] = item
            sim_results['mean_demand'] = mean_demand
            sim_results['std_demand'] = std_demand
            sim_results['safety_stock'] = safety_stock
            sim_results['reorder_point'] = s
            sim_results['order_up_to'] = S
            
            results.append(sim_results)
        
        self.inventory_results = pd.DataFrame(results)
        
        print("\nInventory Simulation Results:")
        print(self.inventory_results[['item_id', 'service_level', 'avg_inventory', 
                                     'total_cost', 'reorder_point', 'order_up_to']].round(2))
        
        self._visualize_inventory_simulation(self.top_items[0])
        
        return self.inventory_results
    
    def _run_inventory_simulation(self, forecast, s, S, lead_time, h_cost, s_cost, o_cost):
        n_days = len(forecast)
        inventory = S
        total_holding_cost = 0
        total_stockout_cost = 0
        total_ordering_cost = 0
        stockouts = 0
        orders = 0
        pending_orders = []
        inventory_levels = []
        
        for day in range(n_days):
            # Receive pending orders
            delivered = [o for o in pending_orders if o['arrival'] <= day]
            for order in delivered:
                inventory += order['quantity']
                pending_orders.remove(order)
            
            # Meet demand
            demand = max(0, forecast[day])  
            if inventory >= demand:
                inventory -= demand
                units_sold = demand
            else:
                units_sold = inventory
                stockouts += 1
                total_stockout_cost += s_cost * (demand - inventory)
                inventory = 0
            
            # Check reorder
            if inventory <= s and not pending_orders:
                order_qty = S - inventory
                orders += 1
                total_ordering_cost += o_cost
                pending_orders.append({
                    'quantity': order_qty,
                    'arrival': day + lead_time
                })
            
            total_holding_cost += h_cost * inventory
            inventory_levels.append(inventory)
        
        return {
            'service_level': 1 - stockouts / n_days,
            'avg_inventory': np.mean(inventory_levels),
            'total_stockouts': stockouts,
            'total_orders': orders,
            'holding_cost': total_holding_cost,
            'stockout_cost': total_stockout_cost,
            'ordering_cost': total_ordering_cost,
            'total_cost': total_holding_cost + total_stockout_cost + total_ordering_cost,
            'inventory_levels': inventory_levels
        }
    
    def _visualize_inventory_simulation(self, item):
        item_results = self.inventory_results[self.inventory_results['item_id'] == item].iloc[0]
        inventory_levels = item_results['inventory_levels']
        
        plt.figure(figsize=(12, 6))
        days = range(len(inventory_levels))
        
        plt.plot(days, inventory_levels, 'b-', linewidth=2, label='Inventory Level')
        plt.axhline(y=item_results['reorder_point'], color='orange', linestyle='--', 
                   label=f"Reorder Point (s={item_results['reorder_point']:.0f})")
        plt.axhline(y=item_results['order_up_to'], color='green', linestyle='--', 
                   label=f"Order-up-to Level (S={item_results['order_up_to']:.0f})")
        
        plt.title(f'Inventory Simulation for {item}')
        plt.xlabel('Day')
        plt.ylabel('Inventory Units')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'inventory_simulation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("\nInventory simulation saved to outputs/inventory_simulation.png")

def main():
    print("="*60)
    print("Using LightGBM with Tweedie Loss")
    print("="*60)
    
    pipeline = M5ForecastingPipeline(
        store_id='CA_1',
        n_items=5,
        test_store_id='CA_2'
    )
    
    try:
        pipeline.load_and_preprocess_data()
        pipeline.exploratory_analysis()
        pipeline.create_cv_splits()
        
        pipeline.train_recursive_model()
        pipeline.train_direct_models(horizon=14)
        
        pipeline.evaluate_models()
        
        pipeline.make_final_predictions()
        pipeline.visualize_forecasts()
        
        pipeline.simulate_inventory_advanced()
        
        print("\n" + "="*60)
        print("Pipeline executed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n  ERROR: {str(e)}")
        raise e
    
    return pipeline

if __name__ == "__main__":
    if not os.path.exists('data'):
        print("  ERROR: 'data' folder not found!")
        print("  Current directory:", os.getcwd())
        sys.exit(1)
        
    pipeline = main()