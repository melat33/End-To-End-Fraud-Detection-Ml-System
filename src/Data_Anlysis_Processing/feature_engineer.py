# task1-data-preprocessing/src/Data_Anlysis_Processing/feature_engineer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection.
    Creates time-based, behavioral, and risk-scoring features.
    """
    
    def __init__(self, config_path: str = "config/config_data_analysis.yaml"):
        """Initialize with configuration."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['feature_engineering']
        self.time_config = self.feature_config['time_features']
        self.behavior_config = self.feature_config['behavioral_features']
        
    def engineer_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for fraud data.
        
        Args:
            df: Fraud dataframe with country mapping
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering for fraud data...")
        
        df_copy = df.copy()
        
        # 1. Time-based features
        df_copy = self._create_time_features(df_copy)
        
        # 2. Behavioral features
        df_copy = self._create_behavioral_features(df_copy)
        
        # 3. Risk-scoring features
        df_copy = self._create_risk_features(df_copy)
        
        # 4. Interaction features
        df_copy = self._create_interaction_features(df_copy)
        
        # 5. Statistical features
        df_copy = self._create_statistical_features(df_copy)
        
        logger.info(f"Feature engineering complete. Added {len(df_copy.columns) - len(df.columns)} new features.")
        logger.info(f"Total features: {len(df_copy.columns)}")
        
        return df_copy
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        # Extract time components
        if self.time_config['extract_hour']:
            df['purchase_hour'] = df['purchase_time'].dt.hour
            df['signup_hour'] = df['signup_time'].dt.hour
        
        if self.time_config['extract_day_of_week']:
            df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek
            df['signup_dayofweek'] = df['signup_time'].dt.dayofweek
        
        if self.time_config['extract_month']:
            df['purchase_month'] = df['purchase_time'].dt.month
            df['signup_month'] = df['signup_time'].dt.month
        
        # Time since signup (CRITICAL FEATURE from YOUR data)
        if self.time_config['time_since_signup']:
            df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
            
            # Create flags based on time differences
            df['is_immediate_purchase'] = (df['time_since_signup_hours'] < 1).astype(int)
            df['is_same_day_purchase'] = (df['time_since_signup_hours'] < 24).astype(int)
            
            # Log insights from YOUR data
            immediate_fraud = df[(df['is_immediate_purchase'] == 1) & (df['class'] == 1)]
            if len(immediate_fraud) > 0:
                logger.warning(f"🚨 Found {len(immediate_fraud)} fraud cases in immediate purchases!")
        
        # Time of day features
        if self.time_config['is_night']:
            night_start, night_end = self.time_config['night_hours']
            df['is_night_transaction'] = ((df['purchase_hour'] >= night_start) | 
                                         (df['purchase_hour'] < night_end)).astype(int)
        
        if self.time_config['is_weekend']:
            df['is_weekend'] = df['purchase_dayofweek'].isin([5, 6]).astype(int)
        
        # Business hours feature
        df['is_business_hours'] = ((df['purchase_hour'] >= 9) & (df['purchase_hour'] <= 17)).astype(int)
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user behavior features."""
        # Sort by user and time for rolling calculations
        df = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)
        
        # Transaction velocity (CRITICAL for fraud detection)
        if self.behavior_config['transaction_velocity']:
            for window in self.behavior_config['velocity_window_hours']:
                # Calculate transactions per user in time window
                df[f'transactions_last_{window}h'] = df.groupby('user_id').rolling(
                    f'{window}h', on='purchase_time'
                )['user_id'].count().values
                
                # Calculate spending velocity
                df[f'spending_last_{window}h'] = df.groupby('user_id').rolling(
                    f'{window}h', on='purchase_time'
                )['purchase_value'].sum().values
        
        # Device risk score
        if self.behavior_config['device_risk_score']:
            # Number of unique users per device
            device_user_counts = df.groupby('device_id')['user_id'].nunique()
            df['users_per_device'] = df['device_id'].map(device_user_counts)
            
            # Device age (time since first seen)
            device_first_seen = df.groupby('device_id')['purchase_time'].min()
            df['device_age_days'] = (df['purchase_time'] - df['device_id'].map(device_first_seen)).dt.days
            
            # Device risk score
            df['device_risk_score'] = np.where(
                df['users_per_device'] > 3, 5, 0
            ) + np.where(
                df['device_age_days'] < 1, 3, 0
            )
        
        # User purchase statistics
        if self.behavior_config['user_purchase_stats']:
            user_stats = df.groupby('user_id').agg({
                'purchase_value': ['mean', 'std', 'min', 'max', 'count'],
                'time_since_signup_hours': 'mean'
            }).round(2)
            
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
            
            # Merge with original data
            for col in user_stats.columns:
                df[f'user_{col}'] = df['user_id'].map(user_stats[col])
            
            # Deviation from user's normal behavior
            df['purchase_deviation'] = (df['purchase_value'] - df['user_purchase_value_mean']) / df['user_purchase_value_std']
            df['purchase_deviation'] = df['purchase_deviation'].fillna(0)
            
            # Flag for unusual purchases
            df['is_unusual_purchase'] = (abs(df['purchase_deviation']) > 2).astype(int)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-scoring features."""
        # Browser risk (some browsers more associated with fraud)
        browser_risk = {
            'Chrome': 1,
            'Safari': 1,
            'Firefox': 2,
            'IE': 3,
            'Opera': 4  # From YOUR data: Opera used in fraud case
        }
        df['browser_risk'] = df['browser'].map(browser_risk).fillna(5)
        
        # Source risk
        source_risk = {
            'Direct': 1,
            'SEO': 2,
            'Ads': 3
        }
        df['source_risk'] = df['source'].map(source_risk).fillna(4)
        
        # Age risk (younger users might be higher risk)
        df['age_risk'] = np.where(df['age'] < 25, 3, 
                                 np.where(df['age'] > 60, 2, 1))
        
        # Purchase value risk
        df['amount_risk'] = pd.cut(df['purchase_value'], 
                                  bins=[0, 10, 50, 100, 500, float('inf')],
                                  labels=[1, 2, 3, 4, 5])
        
        # Country risk (from geolocation)
        if 'country' in df.columns:
            country_risk = self._calculate_country_risk(df)
            df['country_risk'] = df['country'].map(country_risk).fillna(5)
        
        # Composite risk score
        risk_factors = ['browser_risk', 'source_risk', 'age_risk', 
                       'amount_risk', 'country_risk', 'device_risk_score']
        
        available_factors = [f for f in risk_factors if f in df.columns]
        
        if available_factors:
            df['composite_risk_score'] = df[available_factors].sum(axis=1)
            
            # Normalize to 0-100 scale
            max_score = df['composite_risk_score'].max()
            if max_score > 0:
                df['composite_risk_score'] = (df['composite_risk_score'] / max_score) * 100
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # Browser-source interaction
        df['browser_source'] = df['browser'] + '_' + df['source']
        
        # Time-browser interaction
        df['hour_browser'] = df['purchase_hour'].astype(str) + '_' + df['browser']
        
        # Country-browser interaction
        if 'country' in df.columns:
            df['country_browser'] = df['country'] + '_' + df['browser']
        
        # Age-group and purchase category
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 45, 55, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        df['purchase_category'] = pd.cut(df['purchase_value'],
                                       bins=[0, 20, 50, 100, 200, float('inf')],
                                       labels=['micro', 'small', 'medium', 'large', 'xlarge'])
        
        # Create interaction between age group and purchase category
        df['age_purchase_interaction'] = df['age_group'].astype(str) + '_' + df['purchase_category'].astype(str)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        # Z-scores for purchase value by user
        if 'user_purchase_value_mean' in df.columns and 'user_purchase_value_std' in df.columns:
            df['purchase_z_score'] = df['purchase_deviation']  # Already calculated
        
        # Rolling statistics
        if 'purchase_value' in df.columns and 'purchase_time' in df.columns:
            # Sort for rolling
            df = df.sort_values(['user_id', 'purchase_time'])
            
            # Rolling mean and std
            df['rolling_mean_3'] = df.groupby('user_id')['purchase_value'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df['rolling_std_3'] = df.groupby('user_id')['purchase_value'].transform(
                lambda x: x.rolling(3, min_periods=1).std()
            )
        
        # Time between transactions
        df['time_since_last_txn'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)
        
        # Flag for rapid successive transactions
        df['is_rapid_sequence'] = (df['time_since_last_txn'] < 0.5).astype(int)
        
        return df
    
    def _calculate_country_risk(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk scores for countries based on fraud rates."""
        country_stats = df.groupby('country').agg(
            total=('class', 'count'),
            fraud=('class', 'sum')
        )
        
        country_stats['fraud_rate'] = (country_stats['fraud'] / country_stats['total']) * 100
        
        # Normalize to 1-10 scale
        max_rate = country_stats['fraud_rate'].max()
        min_rate = country_stats['fraud_rate'].min()
        
        if max_rate > min_rate:
            country_stats['risk_score'] = 1 + ((country_stats['fraud_rate'] - min_rate) / 
                                             (max_rate - min_rate)) * 9
        else:
            country_stats['risk_score'] = 5
        
        return country_stats['risk_score'].to_dict()
    
    def engineer_creditcard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for credit card data.
        
        Args:
            df: Credit card dataframe
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering for credit card data...")
        
        df_copy = df.copy()
        
        # 1. Amount transformations
        if self.config['data_cleaning']['creditcard_data']['amount_log_transform']:
            df_copy['log_amount'] = np.log1p(df_copy['Amount'])
            df_copy['amount_sqrt'] = np.sqrt(df_copy['Amount'])
        
        # 2. Time-based features
        if 'Time' in df_copy.columns:
            # Extract time of day (assuming Time is seconds from first transaction)
            df_copy['transaction_hour'] = (df_copy['Time'] % 86400) / 3600
            
            # Create time-based features
            df_copy['is_night'] = ((df_copy['transaction_hour'] >= 0) & 
                                  (df_copy['transaction_hour'] < 6)).astype(int)
            df_copy['is_business_hours'] = ((df_copy['transaction_hour'] >= 9) & 
                                           (df_copy['transaction_hour'] <= 17)).astype(int)
        
        # 3. Interaction features between Amount and V features
        v_columns = [col for col in df_copy.columns if col.startswith('V')]
        
        for v_col in v_columns[:5]:  # Limit to first 5 for efficiency
            df_copy[f'amount_{v_col}_interaction'] = df_copy['Amount'] * df_copy[v_col]
        
        # 4. Statistical features for V columns
        df_copy['v_features_mean'] = df_copy[v_columns].mean(axis=1)
        df_copy['v_features_std'] = df_copy[v_columns].std(axis=1)
        df_copy['v_features_max'] = df_copy[v_columns].max(axis=1)
        df_copy['v_features_min'] = df_copy[v_columns].min(axis=1)
        
        # 5. PCA component ratios (capture relationships)
        if 'V1' in df_copy.columns and 'V2' in df_copy.columns:
            df_copy['v1_v2_ratio'] = df_copy['V1'] / (df_copy['V2'] + 1e-10)
        
        logger.info(f"Added {len(df_copy.columns) - len(df.columns)} features to credit card data")
        
        return df_copy
    
    def create_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create features for all datasets.
        
        Args:
            data_dict: Dictionary containing all datasets
            
        Returns:
            Dictionary with engineered datasets
        """
        result = {}
        
        if 'fraud_data' in data_dict and data_dict['fraud_data'] is not None:
            result['fraud_data'] = self.engineer_fraud_features(data_dict['fraud_data'])
        
        if 'creditcard_data' in data_dict and data_dict['creditcard_data'] is not None:
            result['creditcard_data'] = self.engineer_creditcard_features(data_dict['creditcard_data'])
        
        # Copy other datasets unchanged
        for key, df in data_dict.items():
            if key not in result:
                result[key] = df
        
        return result